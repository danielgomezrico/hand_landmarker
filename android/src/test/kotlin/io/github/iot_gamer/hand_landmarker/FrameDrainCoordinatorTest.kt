package io.github.iot_gamer.hand_landmarker

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * JVM unit tests for [FrameDrainCoordinator]. Android-free — no Bitmap, no MediaPipe, no JNI.
 *
 * Proves:
 *  (a) Latest-frame-wins: when N frames are enqueued before drain runs, only the newest
 *      is processed; older ones are recycled to the free pool.
 *  (b) No lost wakeup: a frame arriving while drain is processing causes the drain to loop
 *      and pick up the new frame (wip-counter guarantee).
 *  (c) Free-pool reuse: recycled slots (stale + processed) are returned to the pool;
 *      no unbounded allocation across frames.
 */
internal class FrameDrainCoordinatorTest {

    // Minimal slot type — no Android deps
    private class Slot(val id: Int)

    // ── (a) latest-frame-wins ──────────────────────────────────────────────────

    @Test
    fun singleFrame_drainProcessesIt() {
        val coord = FrameDrainCoordinator<Slot>()
        val processed = mutableListOf<Int>()

        val shouldSubmit = coord.enqueue(Slot(1))
        assertTrue(shouldSubmit, "first enqueue must request a drain task (wip was 0)")

        coord.drain { processed.add(it.id) }
        assertEquals(listOf(1), processed)
    }

    @Test
    fun threeEnqueuedBeforeDrain_onlyLatestProcessed() {
        val coord = FrameDrainCoordinator<Slot>()
        val processed = mutableListOf<Int>()

        // Enqueue 3 frames without running drain
        coord.enqueue(Slot(1))
        coord.enqueue(Slot(2))
        coord.enqueue(Slot(3))

        // One drain pass — must process only the latest (slot 3)
        coord.drain { processed.add(it.id) }

        assertEquals(listOf(3), processed, "only the latest frame must be processed")
    }

    @Test
    fun threeEnqueuedBeforeDrain_olderTwoRecycledToFreePool() {
        val coord = FrameDrainCoordinator<Slot>()
        val s1 = Slot(1); val s2 = Slot(2); val s3 = Slot(3)

        coord.enqueue(s1)   // s1 pending
        coord.enqueue(s2)   // s2 pending, s1 → freePool
        coord.enqueue(s3)   // s3 pending, s2 → freePool

        // Before drain: free pool should hold s1 and s2
        val recycled = buildSet { repeat(2) { add(coord.pollFree()?.id) } }
        assertEquals(setOf(1, 2), recycled, "stale frames must be recycled to the free pool")
        assertNull(coord.pollFree(), "free pool must be empty after retrieving both stale slots")
    }

    @Test
    fun processedSlot_recycledToFreePoolAfterDrain() {
        val coord = FrameDrainCoordinator<Slot>()
        val s = Slot(7)

        coord.enqueue(s)
        coord.drain { /* process */ }

        // After drain the processed slot should be back in the pool
        val recycled = coord.pollFree()
        assertEquals(7, recycled?.id, "processed slot must be offered back to the free pool")
    }

    // ── (b) no lost wakeup ────────────────────────────────────────────────────

    @Test
    fun frameArrivingDuringDrain_isPickedUpByDrainLoop() {
        // Simulates: drain picks up frame 1, then frame 2 arrives *while drain is running*
        // (inside the process callback). The wip counter ensures drain loops to pick up frame 2.
        val coord = FrameDrainCoordinator<Slot>()
        val processed = mutableListOf<Int>()

        coord.enqueue(Slot(1))

        coord.drain { slot ->
            processed.add(slot.id)
            if (slot.id == 1) {
                // Frame 2 arrives mid-drain; wip goes 0→1→2 then back 2→1 as drain loops
                val shouldSubmit = coord.enqueue(Slot(2))
                // drain is already running → wip was 1 (not 0) → no new task needed
                assertTrue(!shouldSubmit, "enqueue during drain must NOT request a new drain task")
            }
        }

        assertEquals(listOf(1, 2), processed,
            "drain must loop to process frame 2 that arrived during frame 1 processing")
    }

    @Test
    fun secondEnqueueWhileDrainIdle_requestsNewDrainTask() {
        // After one drain completes (wip→0), a new enqueue must return true so the caller
        // submits another drain task — otherwise the new frame would be silently dropped.
        val coord = FrameDrainCoordinator<Slot>()

        // First cycle
        coord.enqueue(Slot(1))
        coord.drain { /* process */ }

        // Second cycle — wip is back to 0
        val shouldSubmit = coord.enqueue(Slot(2))
        assertTrue(shouldSubmit, "enqueue after an idle drain must request a new drain task")
    }

    // ── (c) free-pool reuse ───────────────────────────────────────────────────

    @Test
    fun pollFreeAndOfferFree_returnSameInstance() {
        val coord = FrameDrainCoordinator<Slot>()
        val original = Slot(42)
        coord.offerFree(original)

        val borrowed = coord.pollFree()
        assertEquals(42, borrowed?.id)

        // Return it, then borrow again
        if (borrowed != null) coord.offerFree(borrowed)
        val reused = coord.pollFree()

        assertTrue(reused === original, "slot must be the same object instance — no spurious allocation")
    }

    @Test
    fun processedAndStaleSlots_allReturnToFreePool() {
        // 3 enqueues + 1 drain: 2 stale → freePool via enqueue; 1 processed → freePool via drain.
        // After drain the pool must hold all 3 slots.
        val coord = FrameDrainCoordinator<Slot>()
        coord.enqueue(Slot(1))
        coord.enqueue(Slot(2))
        coord.enqueue(Slot(3))

        coord.drain { /* process slot 3 */ }

        // Pool should now contain all three (stale 1 & 2 from enqueue; processed 3 from drain)
        val ids = buildSet { repeat(3) { add(coord.pollFree()?.id) } }
        assertEquals(setOf(1, 2, 3), ids, "all three slots must be in the free pool after drain")
        assertNull(coord.pollFree(), "pool must be empty after retrieving all three slots")
    }

    // ── (e) exception safety ────────────────────────────────────────────────────────────────

    @Test
    fun processThrows_slotReturnedToFreePool_wipReset_exceptionPropagates() {
        // Proves P0 fix: a throwing process() leaves the coordinator in a consistent state.
        val coord = FrameDrainCoordinator<Slot>()
        val slot = Slot(99)
        assertTrue(coord.enqueue(slot), "enqueue must signal that a drain is needed (wip 0→1)")

        val boom = RuntimeException("synthetic boom")
        var caught: Throwable? = null
        try {
            coord.drain { throw boom }
        } catch (t: Throwable) {
            caught = t
        }

        // Exception propagates to the caller
        assertSame(boom, caught, "the original exception must propagate out of drain")
        // Slot is returned to free pool despite the throw
        assertSame(slot, coord.pollFree(), "slot must be in free pool after a throwing process()")
        // wip is back to 0: next enqueue must return true, proving the pipeline is still live
        assertTrue(coord.enqueue(Slot(100)), "next enqueue must return true after recover (wip was reset)")
    }

    @Test
    fun cancelPending_slotReturnedToFreePool_wipReset() {
        // Proves P2 fix: RejectedExecutionException path leaves the coordinator leak-free.
        val coord = FrameDrainCoordinator<Slot>()
        val slot = Slot(77)
        // Simulate: enqueue returned true (wip→1), but drain task was then rejected by the executor.
        assertTrue(coord.enqueue(slot))

        // Handler calls cancelPending to clean up the stranded slot.
        coord.cancelPending()

        // Slot must be in the free pool, not stranded in pending.
        assertSame(slot, coord.pollFree(), "stranded slot must be returned to free pool by cancelPending")
        // wip must be 0 so the next enqueue schedules a new drain.
        assertTrue(coord.enqueue(Slot(78)), "next enqueue must return true after cancelPending (wip was reset)")
    }

    @Test
    fun processThrowsWithConcurrentEnqueue_pipelineStrandedUntilCancelPending() {
        // Proves the residual P0: when a frame is enqueued WHILE process() is throwing, the
        // wip decrement-by-missed leaves wip>=1, so a plain throw does NOT recover the pipeline.
        // drainTask's catch must call cancelPending() for unconditional recovery.
        val coord = FrameDrainCoordinator<Slot>()
        val a = Slot(1)
        assertTrue(coord.enqueue(a), "first enqueue schedules a drain (wip 0→1)")

        // process(a) simulates a camera frame (b) arriving mid-flight, then throws.
        val b = Slot(2)
        var caught: Throwable? = null
        try {
            coord.drain {
                coord.enqueue(b) // concurrent frame: wip 1→2, returns false
                throw RuntimeException("boom during burst")
            }
        } catch (t: Throwable) {
            caught = t
        }
        assertNotNull(caught, "the process exception must propagate out of drain")

        // Stranded: wip is left >= 1, so a fresh enqueue does NOT schedule a drain.
        val c = Slot(3)
        assertFalse(
            coord.enqueue(c),
            "with wip stranded >= 1, enqueue must return false — pipeline would be permanently dead",
        )

        // Recovery: cancelPending resets wip and recycles the stranded pending slot.
        coord.cancelPending()
        assertTrue(coord.enqueue(Slot(4)), "after cancelPending the next enqueue reschedules a drain (wip reset to 0)")

        // No slot leaked: a (processed), b and c (stranded/displaced) all returned to the pool.
        val ids = buildSet { repeat(3) { add(coord.pollFree()?.id) } }
        assertEquals(setOf(1, 2, 3), ids, "all stranded/processed slots must be back in the free pool — none leaked")
    }
}
