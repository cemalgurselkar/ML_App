import subprocess
import time

flask_process = subprocess.Popen(["python", "model_api.py"])
time.sleep(2)
subprocess.run(["streamlit", "run", "app2.py"])
flask_process.terminate()