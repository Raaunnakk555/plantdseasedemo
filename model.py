import io
import os
from PIL import Image
import numpy as np

# Env-switchable artifact loader (to be replaced with your trained model)
MODEL_KIND = os.getenv("MODEL_KIND", "stub")   # stub | keras | savedmodel | tflite
MODEL_PATH = os.getenv("MODEL_PATH", "")

CLASS_NAMES = [
    "Healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Potato___Early_blight",
    "Apple___Black_rot",
]

_loaded = None
_interpreter = None
_input_details = None
_output_details = None

def _load_if_needed():
    global _loaded, _interpreter, _input_details, _output_details
    if _loaded is not None:
        return
    if MODEL_KIND == "tflite":
        import tensorflow as tf  # requires tf in your env
        _interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        _interpreter.allocate_tensors()
        _input_details = _interpreter.get_input_details()
        _output_details = _interpreter.get_output_details()
        _loaded = True
    elif MODEL_KIND in ("keras", "savedmodel"):
        import tensorflow as tf  # requires tf in your env
        if MODEL_KIND == "keras":
            _loaded = tf.keras.models.load_model(MODEL_PATH)
        else:
            _loaded = tf.keras.models.load_model(MODEL_PATH)
    else:
        _loaded = "stub"

def preprocess(img: Image.Image, target_size=(224, 224)) -> np.ndarray:
    img = img.convert("RGB").resize(target_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr

def _predict_stub(arr: np.ndarray):
    # A tiny heuristic: more green -> assume Healthy; more brownish -> Late blight
    img = (arr[0] * 255).astype(np.uint8)
    mean_rgb = img.mean(axis=(0,1))
    r,g,b = mean_rgb.tolist()
    # naive rules just for demo visuals
    if g > r and g > b:
        label = "Healthy"
        conf = 0.72
    elif r < 90 and g < 90 and b < 90:
        label = "Tomato___Late_blight"
        conf = 0.68
    else:
        # random-ish pick with deterministic hash
        idx = int((r + 2*g + 3*b) % (len(CLASS_NAMES)-1)) + 1
        label = CLASS_NAMES[idx]
        conf = 0.61
    return label, conf

def predict(image_bytes: bytes):
    _load_if_needed()
    img = Image.open(io.BytesIO(image_bytes))
    arr = preprocess(img)
    if MODEL_KIND == "stub":
        label, conf = _predict_stub(arr)
        return label, float(conf)
    elif MODEL_KIND == "tflite":
        import numpy as np
        global _interpreter, _input_details, _output_details
        inp = arr.astype(_input_details[0]["dtype"])
        _interpreter.set_tensor(_input_details[0]["index"], inp)
        _interpreter.invoke()
        out = _interpreter.get_tensor(_output_details[0]["index"])
        idx = int(np.argmax(out[0]))
        conf = float(out[0][idx])
        return CLASS_NAMES[idx], conf
    else:
        # keras/savedmodel
        import numpy as np
        logits = _loaded.predict(arr, verbose=0)
        idx = int(np.argmax(logits[0]))
        conf = float(np.max(logits[0]))
        # In real training you will have your own class order; adjust CLASS_NAMES accordingly.
        return CLASS_NAMES[idx], conf
