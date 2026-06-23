# locustfile.py
import random
import uuid
from locust import HttpUser, task, between

class PaymentTerminalSimulator(HttpUser):
    """
    Simulates a fleet of payment terminals globally.
    Each user waits between 0.01 and 0.1 seconds between swipes.
    """
    wait_time = between(0.01, 0.1)

    @task
    def submit_fraud_check(self):
        """
        Generates a randomized payload mimicking the enriched transaction
        data that your Redis Feature Store would output.
        """
        # Generate dummy transaction data
        tx_id = f"tx_{uuid.uuid4().hex[:8]}"
        user_id = f"user_{random.randint(1000, 9999)}"
        amount = round(random.uniform(5.0, 2500.0), 2)
        
        # Historical features
        avg_spend_30d = round(random.uniform(10.0, 500.0), 2)
        risk_score = round(random.uniform(0.0, 1.0), 2)
        
        payload = [{
            "tx_id": tx_id,
            "user_id": user_id,
            "amount": amount,
            "current_city": random.choice(["New York", "London", "Tokyo", "Paris", "Austin"]),
            "avg_spend_30d": avg_spend_30d,
            "risk_score": risk_score,
            "last_location": random.choice(["New York", "London", "Tokyo", "UNKNOWN"])
        }]

        # POST the payload to the inference server
        # We catch the response to ensure we are returning HTTP 200s, not 503s or 413s
        with self.client.post("/predict", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                # Check if we violated the 50ms SLA
                inference_time = response.json()[0].get("inference_time_ms", 0)
                if inference_time > 50.0:
                    response.failure(f"SLA Breach: Took {inference_time:.2f}ms")
                else:
                    response.success()
            else:
                response.failure(f"Failed with status code: {response.status_code}")

# --- Quick Test Instructions ---
# 1. Ensure your Docker Compose cluster is running: `docker-compose up -d`
# 2. Run this script in your terminal: `locust -f locustfile.py`
# 3. Open your browser to http://localhost:8089 to access the Locust dashboard.
# 4. Set "Number of users" to 10,000 and "Spawn rate" to 500 to simulate a massive traffic spike.