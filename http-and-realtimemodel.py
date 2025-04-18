import numpy as np
import tflite_runtime.interpreter as tflite
import time
import requests
from sklearn.preprocessing import MinMaxScaler
from mpu6050 import mpu6050
import RPi.GPIO as GPIO

# Initialize MPU6050 sensor
sensor = mpu6050(0x68)

# Setup GPIO for Alarm
GPIO.setmode(GPIO.BCM)
ALARM_PIN = 17
GPIO.setup(ALARM_PIN, GPIO.OUT)

# Load TFLite model
tflite_model_path = "/home/aartech/Desktop/vibration sensor final/vibration_model.tflite"
interpreter = tflite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input/output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Constants
URL = "https://predictiveai.in/api/receive-data/"
window_size = 100
num_features = 3
threshold = 0.02  # Fault detection threshold

def get_real_time_vibration_data():
    readings = []
    start_time = time.time()
    while len(readings) < window_size:
        accel_data = sensor.get_accel_data()
        readings.append([accel_data['x'], accel_data['y'], accel_data['z']])
    end_time = time.time()
    print(f"⏱️ Data captured in {end_time - start_time:.3f} sec")
    return np.array(readings)

def trigger_alarm():
    GPIO.output(ALARM_PIN, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(ALARM_PIN, GPIO.LOW)

def send_to_server(x, y, z):
    payload = {"x": x, "y": y, "z": z}
    try:
        response = requests.post(URL, json=payload)
        print(f"🌐 Sent to Server: {payload} | Status: {response.status_code}")
    except Exception as e:
        print("❌ Failed to send data:", e)

def detect_fault_and_send():
    vibration_data = get_real_time_vibration_data()

    # Normalize data
    scaler = MinMaxScaler()
    vibration_data = scaler.fit_transform(vibration_data)

    # Reshape for model
    X_test = vibration_data.reshape((1, window_size, num_features)).astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], X_test)
    interpreter.invoke()
    reconstructed = interpreter.get_tensor(output_details[0]['index'])

    # Error calculation
    error = np.mean((X_test - reconstructed) ** 2)
    print(f"🔍 Reconstruction Error: {error:.6f}")

    # Last x, y, z reading
    x, y, z = sensor.get_accel_data().values()

    # Fault check
    if error > threshold:
        print("🚨 Fault Detected!")
        trigger_alarm()
    else:
        print("✅ Machine is Healthy.")

    # Send x, y, z to Django server
    send_to_server(x, y, z)

# Main loop
try:
    while True:
        print("\n🔄 Checking machine health...")
        loop_start = time.time()
        detect_fault_and_send()
        loop_time = time.time() - loop_start
        print(f"⚙️ Loop completed in {loop_time:.3f} sec")
        if loop_time < 1.0:
            time.sleep(1.0 - loop_time)  # Ensure ~1 second per loop
except KeyboardInterrupt:
    print("🛑 Monitoring Stopped by User.")
    GPIO.cleanup()

