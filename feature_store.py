# feature_store.py
import asyncio
import time

class RedisFeatureStoreMock:
    def __init__(self):
        """
        Simulates an ultra-fast Redis in-memory key-value cache.
        In production, this would be a connection to a Redis cluster.
        """
        # Our "RAM". Lookups in a Python dict (hash map) are O(1) time complexity.
        self._memory_cache = {
            "user_991": {"avg_spend_30d": 45.50, "risk_score": 0.1, "last_location": "New York"},
            "user_442": {"avg_spend_30d": 1200.00, "risk_score": 0.8, "last_location": "London"},
            "user_773": {"avg_spend_30d": 15.00, "risk_score": 0.05, "last_location": "Tokyo"}
        }

    async def get_features(self, user_id):
        """
        Asynchronously retrieves the user's pre-calculated historical features.
        Simulates network latency to the Redis cluster (e.g., 2ms).
        """
        # Simulating the ultra-fast network hop to a Redis node
        await asyncio.sleep(0.002) 
        
        # O(1) retrieval
        return self._memory_cache.get(user_id, self._default_features())

    def _default_features(self):
        """Fallback features for brand new users with no history."""
        return {"avg_spend_30d": 0.0, "risk_score": 0.5, "last_location": "UNKNOWN"}

    async def enrich_transaction(self, transaction):
        """
        Takes a raw incoming transaction and merges it with the historical features.
        This creates the final 'Feature Vector' that the neural network requires.
        """
        user_id = transaction.get("user_id")
        
        # Fetch the historical context from RAM
        start_time = time.time()
        historical_features = await self.get_features(user_id)
        latency_ms = (time.time() - start_time) * 1000
        
        # Merge the raw data with the historical data
        enriched_vector = {**transaction, **historical_features}
        
        return enriched_vector, latency_ms


# --- Quick Test ---
async def main():
    print("⚡ Booting In-Memory Feature Store (Redis Mock)...")
    feature_store = RedisFeatureStoreMock()
    
    # Simulate raw transactions hitting the ingestion engine
    raw_transactions = [
        {"tx_id": "tx_001", "user_id": "user_991", "amount": 50.00, "current_city": "New York"},
        {"tx_id": "tx_002", "user_id": "user_442", "amount": 5000.00, "current_city": "Paris"}, # High risk anomaly!
        {"tx_id": "tx_003", "user_id": "user_new", "amount": 20.00, "current_city": "Austin"}
    ]
    
    print("\n🔄 Enriching incoming transaction stream...\n")
    
    for tx in raw_transactions:
        enriched_data, latency = await feature_store.enrich_transaction(tx)
        print(f"✅ Enriched {tx['tx_id']} for {tx['user_id']} | Fetch Latency: {latency:.2f}ms")
        print(f"   [Model Input Vector]: {enriched_data}\n")

if __name__ == "__main__":
    asyncio.run(main())