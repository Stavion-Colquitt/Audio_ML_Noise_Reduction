import sounddevice as sd

devices = sd.query_devices()
for i, device in enumerate(devices):
    print(f"{i}: {device['name']} - inputs: {device['max_input_channels']} outputs: {device['max_output_channels']}")
