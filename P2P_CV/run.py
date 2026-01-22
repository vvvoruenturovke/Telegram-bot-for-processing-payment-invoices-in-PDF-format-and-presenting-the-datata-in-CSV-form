import subprocess
import time
import requests


def run_api():
    print("🚀 Starting API...")
    subprocess.Popen(
        ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8030"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

def wait_api():
    for _ in range(25):
        try:
            r = requests.get("http://localhost:8030/health", timeout=2)
            if r.status_code == 200:
                print("API готово!")
                return True
        except:
            pass
        print("Ждем API...")
        time.sleep(5)
    return False

def run_bot():
    from bot import NeuralInvoiceBot
    NeuralInvoiceBot().run()

if __name__ == "__main__":
    run_api()
    if wait_api():
        run_bot()
    else:
        print("API не запустилось")