package io.github.iot_gamer.hand_landmarker

import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

/**
 * Pure, Android-free, JVM-unit-testable latest-frame-wins drain coordinator.
 *
 * Encapsulates the wip-counter + atomic-pending + free-pool pattern
 * (RxJava-style single-threaded drain, latest wins).
 *
 * Thread contract:
 *  - [enqueue] is called by the producer thread. Returns true iff the caller must submit
 *    a new drain task (wip transitioned 0→1, meaning no drain is currently running).
 *  - [drain] runs the canonical wip-counter loop on a single designated worker thread:
 *    processes the latest pending slot and loops if new frames arrived during processing,
 *    guaranteeing no lost wakeup and no unbounded task queue growth.
 *  - [pollFree] / [offerFree] let the producer borrow/return idle slot objects.
 *
 * Correctness properties (proved by [FrameDrainCoordinatorTest]):
 *  (a) Latest-frame-wins: stale pending frames are recycled to the free pool by [enqueue].
 *  (b) No lost wakeup: a frame arriving while drain is running causes the drain to loop.
 *  (c) Free-pool reuse: processed + stale slots are returned to the pool; no allocation growth.
 */
internal class FrameDrainCoordinator<T> {
    private val pending = AtomicReference<T?>(null)
    private val wip = AtomicInteger(0)
    private val freePool = ConcurrentLinkedQueue<T>()

    /** Borrow a slot from the free pool; returns null if the pool is empty. */
    fun pollFree(): T? = freePool.poll()

    /** Return a slot to the free pool (called by producer after discovering pool is non-empty
     *  or by [drain] after processing / by [enqueue] when a stale slot is displaced). */
    fun offerFree(slot: T) { freePool.offer(slot) }

    /**
     * Enqueue [slot] as the latest pending frame, recycling any previously pending
     * (not yet processed) frame back to the free pool. Returns true iff the caller must
     * submit a drain task (wip was 0 → now 1).
     */
    fun enqueue(slot: T): Boolean {
        val prev = pending.getAndSet(slot)
        if (prev != null) freePool.offer(prev)   // stale frame displaced — recycle it
        return wip.getAndIncrement() == 0
    }

    /**
     * Canonical wip-counter drain loop. Must be called only from the designated single
     * worker thread. Processes only the latest pending slot per pass (older slots were
     * already recycled by [enqueue]). After calling [process], returns the slot to the
     * free pool. Loops if new frames arrived while [process] was running (no lost wakeup).
     *
     * Exception contract: if [process] throws, the slot is STILL returned to the free pool
     * and wip is STILL decremented by [missed] before the exception propagates. NOTE this only
     * lands wip at 0 (so the next [enqueue] reschedules) when no frame arrived during the
     * throwing [process]; if a frame was enqueued concurrently, wip is left >= 1 and the caller
     * MUST call [cancelPending] on the catch path to fully recover the pipeline.
     */
    fun drain(process: (T) -> Unit) {
        var missed = 1
        while (true) {
            val slot = pending.getAndSet(null)
            var thrown: Throwable? = null
            if (slot != null) {
                try {
                    process(slot)
                } catch (t: Throwable) {
                    thrown = t
                } finally {
                    freePool.offer(slot) // ALWAYS recycle — even if process() threw
                }
            }
            missed = wip.addAndGet(-missed) // ALWAYS decrement — wip consistent before rethrow
            if (thrown != null) throw thrown // propagate after state is consistent
            if (missed == 0) return
        }
    }

    /**
     * Emergency cleanup for the executor-rejection race: if [worker.execute(drainTask)]
     * throws [java.util.concurrent.RejectedExecutionException] after [enqueue] returned
     * true, the pending slot is stranded in [pending] and [wip] is non-zero.
     *
     * Call this from inside the RejectedExecutionException catch block to return the
     * slot to the free pool and reset wip to 0, leaving the coordinator leak-free
     * and permanently idle (appropriate — the owning executor is already shut down).
     */
    fun cancelPending() {
        val slot = pending.getAndSet(null)
        if (slot != null) freePool.offer(slot)
        wip.set(0)
    }
}
