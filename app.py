import os
from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU if not configured

import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

def predict(values, dic):
    # diabetes
    if len(values) == 8:
        dic2 = {'NewBMI_Obesity 1': 0, 'NewBMI_Obesity 2': 0, 'NewBMI_Obesity 3': 0, 'NewBMI_Overweight': 0,
                'NewBMI_Underweight': 0, 'NewInsulinScore_Normal': 0, 'NewGlucose_Low': 0,
                'NewGlucose_Normal': 0, 'NewGlucose_Overweight': 0, 'NewGlucose_Secret': 0}

        if dic['BMI'] <= 18.5:
            dic2['NewBMI_Underweight'] = 1
        elif 18.5 < dic['BMI'] <= 24.9:
            pass
        elif 24.9 < dic['BMI'] <= 29.9:
            dic2['NewBMI_Overweight'] = 1
        elif 29.9 < dic['BMI'] <= 34.9:
            dic2['NewBMI_Obesity 1'] = 1
        elif 34.9 < dic['BMI'] <= 39.9:
            dic2['NewBMI_Obesity 2'] = 1
        elif dic['BMI'] > 39.9:
            dic2['NewBMI_Obesity 3'] = 1

        if 16 <= dic['Insulin'] <= 166:
            dic2['NewInsulinScore_Normal'] = 1

        if dic['Glucose'] <= 70:
            dic2['NewGlucose_Low'] = 1
        elif 70 < dic['Glucose'] <= 99:
            dic2['NewGlucose_Normal'] = 1
        elif 99 < dic['Glucose'] <= 126:
            dic2['NewGlucose_Overweight'] = 1
        elif dic['Glucose'] > 126:
            dic2['NewGlucose_Secret'] = 1

        dic.update(dic2)
        values2 = list(map(float, list(dic.values())))

        model = pickle.load(open('models/diabetes.pkl','rb'))
        values = np.asarray(values2)
        return model.predict(values.reshape(1, -1))[0]

    # breast_cancer
    elif len(values) == 9:
        model = pickle.load(open('models/breast_cancers_model.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

    # heart disease
    elif len(values) == 13:
        model = pickle.load(open('models/heart.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

    # kidney disease
    elif len(values) == 24:
        model = pickle.load(open('models/kidney.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

    # liver disease
    elif len(values) == 10:
        model = pickle.load(open('models/liver.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

#
def get_tumor_stage(size_cm, metastasis=None, inv_nodes=None, nodes_threshold=4):
    """
    Determine tumor stage from tumor size (cm), optional metastasis flag and inv_nodes (int or str).
    Returns a string like 'Stage 1' or 'Stage 0 (Tis)'.
    """
    # Defensive parsing
    try:
        # allow None or empty strings
        if size_cm is None or str(size_cm).strip() == "":
            return "Unknown (no size)"
        size = float(size_cm)
    except Exception:
        return "Invalid size"

    # parse metastasis
    metastasis_flag = False
    if metastasis is not None:
        if isinstance(metastasis, str):
            metastasis_flag = metastasis.strip() in ("1", "yes", "true", "True", "Y", "y")
        else:
            metastasis_flag = bool(metastasis)

    # parse inv_nodes
    nodes = None
    try:
        if inv_nodes is not None and str(inv_nodes).strip() != "":
            nodes = int(float(inv_nodes))
    except Exception:
        nodes = None

    # If metastasis present => Stage 4
    if metastasis_flag:
        return "Stage 4"

    # Tis (in-situ)
    if size == 0:
        return "Stage 0 (Tis)"

    # Stage logic
    if size <= 2:
        return "Stage 1"
    if 2 < size <= 5:
        return "Stage 2"
    if size > 5:
        return "Stage 3"

    # fallback
    return "Unknown"

@app.route('/predictPage', methods=['POST'])
# --- Tumor Staging Function ---
def predictPage():
    try:
        import numpy as np
        import pickle
        from flask import request, render_template

        # ✅ Extract and convert form inputs
        #tumor_size = float(request.form['tumor_size'])
        year = float(request.form['year'])
        age = float(request.form['age'])
        menopause = float(request.form['menopause'])
        tumor_size = float(request.form['tumor_size'])
        inv_nodes = float(request.form['inv_nodes'])
        breast = float(request.form['breast'])
        metastasis = float(request.form['metastasis'])
        breast_quadrant = float(request.form['breast_quadrant'])
        history = float(request.form['history'])
        stage = get_tumor_stage(tumor_size)

        # ✅ Combine into NumPy array
        values = np.array([[year, age, menopause, tumor_size,
                            inv_nodes, breast, metastasis,
                            breast_quadrant, history]])

        print("\n🧾 Input Values:", values)

        # ✅ Load trained Logistic Regression model
        model_path = 'models/breast_cancer_modelr.pkl'  # Corrected filename
        model = pickle.load(open(model_path, 'rb'))

        # ✅ Validate model type
        if not hasattr(model, "predict"):
            raise TypeError(f"The file '{model_path}' is not a valid ML model. Recheck the pickle file.")

        # ✅ Predict
        prediction = model.predict(values)[0]

        # ✅ Get prediction probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(values)[0]
            confidence = probs[int(prediction)] * 100
            benign_prob = probs[0] * 100
            malignant_prob = probs[1] * 100
        else:
            confidence = None
            benign_prob = malignant_prob = None

        print(f"🎯 Prediction: {prediction}, Confidence: {confidence}%")

        # ✅ Convert numeric result to descriptive text
        if prediction == 1:
            result = f"🩸 Malignant (Positive Case)"
            description = f"The model predicts a malignant tumor with {confidence:.2f}% confidence."
        else:
            result = f"✅ Benign (Negative Case)"
            description = f"The model predicts a benign tumor with {confidence:.2f}% confidence."

        # ✅ Pass to template
        return render_template(
            'breast_cancer.html',
            prediction_text=result,
            description=description,
            benign_prob=f"{benign_prob:.2f}%" if benign_prob else None,
            malignant_prob=f"{malignant_prob:.2f}%" if malignant_prob else None ,
            year=year,
            age=age,
            menopause=menopause,
            tumor_size=tumor_size,
            inv_nodes=inv_nodes,
            breast=breast,
            metastasis=metastasis,
            tumor_stage=stage,
            breast_quadrant=breast_quadrant,
            history=history
         )
        

    except Exception as e:
        print("⚠️ Error during prediction:", e)
        return render_template(
            'breast_cancer.html',
            prediction_text="⚠️ Please enter valid data.",
            description=str(e)
        )




model = load_model("models/cancer-model.h5", compile=False)
class_labels = ['benign', 'malignant', 'normal']



# ✅ Human-readable messages
label_messages = {
    'benign': 'NEGATIVE — The patient is healthy (benign condition detected).',
    'normal': 'NEGATIVE — The patient is healthy.',
    'malignant': 'POSITIVE — The patient is affected (malignant condition detected).'
}

# ✅ Load model (compile=False for inference)
model = tf.keras.models.load_model("models/cancer-modelS.h5", compile=False)
def get_tumor_stage(size_cm, metastasis=None, inv_nodes=None, nodes_threshold=4):
    """
    Determine tumor stage from tumor size (cm), optional metastasis flag and inv_nodes (int or str).
    Returns a string like 'Stage 1' or 'Stage 0 (Tis)'.
    """
    # Defensive parsing
    try:
        # allow None or empty strings
        if size_cm is None or str(size_cm).strip() == "":
            return "Unknown (no size)"
        size = float(size_cm)
    except Exception:
        return "Invalid size"

    # parse metastasis
    metastasis_flag = False
    if metastasis is not None:
        if isinstance(metastasis, str):
            metastasis_flag = metastasis.strip() in ("1", "yes", "true", "True", "Y", "y")
        else:
            metastasis_flag = bool(metastasis)

    # parse inv_nodes
    nodes = None
    try:
        if inv_nodes is not None and str(inv_nodes).strip() != "":
            nodes = int(float(inv_nodes))
    except Exception:
        nodes = None

    # If metastasis present => Stage 4
    if metastasis_flag:
        return "Stage 4"

    # Tis (in-situ)
    if size == 0:
        return "Stage 0 (Tis)"

    # Stage logic
    if size <= 2:
        return "Stage 1"
    if 2 < size <= 5:
        return "Stage 2"
    if size > 5:
        return "Stage 3"

    # fallback
    return "Unknown"


@app.route("/malariapredict", methods=['POST', 'GET'])
def malariapredictPage():
    pred_label = None
    pred_prob = None
    message = None
    final_message = None

    if request.method == 'POST':
        try:
            # Get uploaded image
            img_file = request.files.get('image')
            if not img_file or img_file.filename == '':
                raise ValueError("No file uploaded")

            # Ensure uploads folder exists
            os.makedirs('uploads', exist_ok=True)
            img_path = os.path.join('uploads', img_file.filename)
            img_file.save(img_path)

            # Load and preprocess image
            img = tf.keras.utils.load_img(img_path, target_size=(128, 128))
            img_array = tf.keras.utils.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = model.predict(img_array)
            print("\n🧠 Raw Predictions:", predictions)  # Debugging line
            pred_index = np.argmax(predictions[0])
            pred_label = class_labels[pred_index]
            pred_prob = round(predictions[0][pred_index] * 100, 2)
            final_message = label_messages[pred_label]

            print(f"🔍 Predicted Index: {pred_index}, Label: {pred_label}, Confidence: {pred_prob}%")

        except Exception as e:
            message = f"Error: {str(e)}. Please upload a valid image."
            return render_template('malaria_predict.html', message=message)

    return render_template('malaria_predict.html',
                           pred_label=pred_label,
                           pred_prob=pred_prob,
                           message=message,
                           final_message=final_message)


@app.route("/pneumoniapredict", methods = ['POST', 'GET'])
def pneumoniapredictPage():
    if request.method == 'POST':
        try:
            img = Image.open(request.files['image']).convert('L')
            img.save("uploads/image.jpg")
            img_path = os.path.join(os.path.dirname(__file__), 'uploads/image.jpg')
            os.path.isfile(img_path)
            img = tf.keras.utils.load_img(img_path, target_size=(128, 128))
            img = tf.keras.utils.img_to_array(img)
            img = np.expand_dims(img, axis=0)

            model = tf.keras.models.load_model("models/pneumonia.h5")
            pred = np.argmax(model.predict(img))
        except:
            message = "Please upload an image"
            return render_template('pneumonia.html', message=message)
    return render_template('pneumonia_predict.html', pred=pred)

if __name__ == '__main__':
    app.run(debug = True)