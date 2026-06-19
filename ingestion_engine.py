# ingestion_engine.py
import asyncio
import time
import uuid

class DropAndCatchQueue:
    def __init__(self, batch_size=16, timeout=0.05):
        """
        Initializes the asynchronous ingestion queue.
        :param batch_size: The strict maximum number of items the GPU will process at once to prevent OOM.
        :param timeout: The maximum time (in seconds) to wait before forcing a batch to process (50ms SLA).
        """
        self.queue = asyncio.Queue()
        self.batch_size = batch_size
        self.timeout = timeout

    async def drop_transaction(self, transaction_data):
        """
        The CPU-bound 'Drop'. Simulates an API endpoint receiving a transaction.
        It pushes the data to the queue and returns instantly to keep TPS high.
        """
        transaction_id = str(uuid.uuid4())[:8]
        await self.queue.put({"id": transaction_id, "data": transaction_data})
        return {"status": "queued", "transaction_id": transaction_id}

    async def catch_and_process(self):
        """
        The GPU-bound 'Catch'. Runs in a continuous background loop, pulling items 
        from the queue until it hits the batch limit or the 50ms timeout.
        """
        while True:
            batch = []
            start_time = time.time()

            # Fill the batch until we hit the size limit OR time runs out
            while len(batch) < self.batch_size:
                try:
                    # Wait for an item, but don't wait longer than our strict SLA timeout
                    time_left = self.timeout - (time.time() - start_time)
                    if time_left <= 0:
                        break
                    
                    item = await asyncio.wait_for(self.queue.get(), timeout=time_left)
                    batch.append(item)
                    self.queue.task_done()
                except asyncio.TimeoutError:
                    break # Timeout hit, force the batch to process immediately

            if batch:
                await self._mock_gpu_inference(batch)

    async def _mock_gpu_inference(self, batch):
        """
        Simulates the actual neural network forward pass.
        Because batch size is strictly capped, VRAM remains stable.
        """
        print(f"🔥 [GPU CATCH] Processing batch of {len(batch)} transactions...")
        # Simulate inference delay
        await asyncio.sleep(0.01) 
        for item in batch:
            print(f"   ✅ Processed TXN: {item['id']}")


# --- Quick Test ---
async def main():
    print("🚀 Booting Real-Time Ingestion Engine...")
    engine = DropAndCatchQueue(batch_size=4, timeout=0.05)
    
    # Start the continuous background GPU processing loop
    asyncio.create_task(engine.catch_and_process())
    
    print("🌊 Simulating massive API traffic spike (10 rapid requests)...")
    
    # Simulate dropping 10 concurrent requests from payment terminals
    tasks = []
    for i in range(10):
        tasks.append(engine.drop_transaction({"amount": 100 + i, "merchant": "Debales.ai"}))
    
    # Wait for all drops to be acknowledged by the API
    results = await asyncio.gather(*tasks)
    print(f"📥 API Acknowledged {len(results)} transactions instantly.")
    
    # Give the background worker a moment to process the batches
    await asyncio.sleep(0.2)
    print("🛑 System idle.")

if __name__ == "__main__":
    asyncio.run(main())