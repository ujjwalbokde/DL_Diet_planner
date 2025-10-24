# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import tensorflow as tf
import pickle
import json
import os
import google.generativeai as genai
from dotenv import load_dotenv






# --- Setup and Model Loading ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="Modular AI Diet Recommendation API")

print("[SERVER STARTUP] Loading all models into memory...")
disease_model = tf.keras.models.load_model("model/disease_predictor_model.h5")
with open("model/label_encoder.pkl", "rb") as f:
    disease_label_encoder = pickle.load(f)
with open("model/symptom_list.json", "r") as f:
    all_symptoms = json.load(f)
with open("model/disease_to_diet_model.json", "r") as f:
    disease_to_diet_map = json.load(f)
print("[SERVER STARTUP] All models loaded successfully.")






# --- Pydantic Input Models ---

# Model for the FIRST API endpoint
class DietTypeInput(BaseModel):
    prompt: str = Field(..., example="I have a constant headache and feel very tired.")
    age: int = Field(..., example=35)
    bmi: float = Field(..., example=22.5)
    gender: str = Field(..., example="Male")

# Model for the SECOND API endpoint
class MealPlanInput(BaseModel):
    diet_type: str = Field(..., example="Low Carb Diet")
    age: int = Field(..., example=35)
    bmi: float = Field(..., example=22.5)
    gender: str = Field(..., example="Male")
    allergies: List[str] = Field(default=[], example=["peanuts", "gluten"])
    dietary_preference: str = Field(..., example="Vegetarian")
    cuisine_preference: str = Field(..., example="North Indian")
    spice_tolerance: str = Field(..., example="Medium")




# --- Helper and Placeholder Functions (Same as before) ---
def extract_symptoms_with_llm(prompt: str, master_symptoms: list) -> list:
    model = genai.GenerativeModel('gemini-pro-latest')
    # ... (rest of the function is the same)
    extraction_prompt = f"""
You are an expert medical symptom extraction bot. Analyze the user's statement and identify every symptom that matches an item from the "Master Symptom List".
Your response MUST BE a valid JSON array of strings from the master list. If no symptoms are found, return an empty array `[]`. Do not add any explanation.

---
**Master Symptom List:**
{json.dumps(master_symptoms)}
---

**User's Statement:**
"{prompt}"
"""
    try:
        response = model.generate_content(extraction_prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        extracted_symptoms = json.loads(cleaned_response)
        valid_symptoms = [s for s in extracted_symptoms if s in master_symptoms]
        return valid_symptoms
    except Exception as e:
        print(f"!!! ERROR in 'extract_symptoms_with_llm': {e}")
        return []





# def generate_final_diet_plan(plan_input: MealPlanInput) -> dict:
#     print(f"\n[API 2] Input to Final Plan Generator -> Diet Type: '{plan_input.diet_type}', Prefs: {plan_input.dietary_preference}, Allergies: {plan_input.allergies}")
    
#     notes = f"A {plan_input.diet_type} for a {plan_input.age}-year-old {plan_input.gender.lower()} who prefers {plan_input.dietary_preference} food."
#     if plan_input.allergies:
#         notes += f" Avoiding: {', '.join(plan_input.allergies)}."

#     final_plan = {
#         "plan_details": { "assigned_diet_type": plan_input.diet_type, "notes": notes },
#         "meals": {
#             "breakfast": {"time": "8:00 AM", "options": ["Generated Option 1", "Generated Option 2"]},
#             "lunch": {"time": "1:00 PM", "options": ["Generated Option 1", "Generated Option 2"]},
#             "snacks": {"time": "4:30 PM", "options": ["Generated Option 1", "Generated Option 2"]},
#             "dinner": {"time": "8:00 PM", "options": ["Generated Option 1", "Generated Option 2"]}
#         }
#     }
#     return final_plan











# --- [UPDATED] COMPLETED Final Diet Plan Generation Function ---
def generate_meal_plan_with_llm(plan_input: MealPlanInput) -> dict:
    """
    Uses the Gemini API to generate a personalized, structured meal plan.
    """
    print(f"\n[API 2] Input to LLM Plan Generator -> Diet Type: '{plan_input.diet_type}', Cuisine: {plan_input.cuisine_preference}, Spice: {plan_input.spice_tolerance}")
    
    model = genai.GenerativeModel('gemini-pro-latest')

    allergies_str = ", ".join(plan_input.allergies) if plan_input.allergies else "None"

    # --- THE PROMPT IS UPDATED WITH THE NEW CONSTRAINT ---
    generation_prompt = f"""
You are an expert Indian nutritionist. Your task is to create a personalized, full-day meal plan based on the user's profile.

**User Profile:**
- **Age:** {plan_input.age}
- **Gender:** {plan_input.gender}
- **BMI:** {plan_input.bmi}
- **Required Diet Type:** "{plan_input.diet_type}"
- **Dietary Preference:** "{plan_input.dietary_preference}"
- **Must Avoid (Allergies):** "{allergies_str}"
- **Cuisine Preference:** "{plan_input.cuisine_preference}"
- **Spice Tolerance:** "{plan_input.spice_tolerance}"

**Instructions:**
1.  Create a meal plan for Breakfast, Lunch, and Dinner.
2.  Provide exactly three distinct options for each meal, labeled "Option 1", "Option 2", and "Option 3".
3.  Each meal option must be a list of strings, where each string is a single food item.
4.  Each meal option list must contain between 2 and 4 food items.
5.  All meal options must be common, healthy Indian dishes that match the user's Cuisine Preference and Spice Tolerance.
6.  Strictly adhere to the user's Dietary Preference and Allergies.
7.  Your response MUST BE a valid JSON object and nothing else. Do not add any explanation or markdown formatting.
8.  The JSON object must follow this exact structure:
    {{
      "Breakfast": {{
        "time": "7:00 AM - 8:30 AM",
        "Option 1": ["Item 1", "Item 2"],
        "Option 2": ["Item 1", "Item 2"],
        "Option 3": ["Item 1", "Item 2", "Item 3"]
      }},
      "Lunch": {{
        "time": "12:30 PM - 1:30 PM",
        "Option 1": ["..."], "Option 2": ["..."], "Option 3": ["..."]
      }},
      "Dinner": {{
        "time": "7:00 PM - 8:00 PM",
        "Option 1": ["..."], "Option 2": ["..."], "Option 3": ["..."]
      }}
    }}
"""
    try:
        response = model.generate_content(generation_prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        final_plan = json.loads(cleaned_response)
        return final_plan
    except (json.JSONDecodeError, Exception) as e:
        print(f"!!! ERROR in 'generate_meal_plan_with_llm': {e}")
        return {"error": "Failed to generate a personalized meal plan.", "details": str(e)}
    







# --- API ENDPOINT 1: Predict Diet Type ---
@app.post("/predict-diet-type")
def predict_diet_type_endpoint(input_data: DietTypeInput):
    try:
        print("\n" + "="*50)
        print(f"🚀 New Request for Diet Type Prediction. User Prompt: '{input_data.prompt}'")
        print("="*50)

        # STEP 1: Extract symptoms
        print(f"\n[STEP 1] Calling LLM to extract symptoms...")
        extracted_symptoms = extract_symptoms_with_llm(input_data.prompt, all_symptoms)
        if not extracted_symptoms:
            raise HTTPException(status_code=400, detail="Could not identify any valid symptoms.")
        print(f"[STEP 1] LLM Response -> Symptoms found: {extracted_symptoms}")

        # STEP 2: Predict disease
        print(f"\n[STEP 2] Calling Disease Predictor model...")
        input_vector = [1 if symptom in extracted_symptoms else 0 for symptom in all_symptoms]
        disease_input = np.array(input_vector).reshape(1, -1)
        disease_probs = disease_model.predict(disease_input)
        predicted_disease_index = np.argmax(disease_probs)
        disease_confidence = float(np.max(disease_probs))
        predicted_disease = disease_label_encoder.inverse_transform([predicted_disease_index])[0]
        print(f"[STEP 2] Model Response -> Predicted Disease: '{predicted_disease}' (Confidence: {disease_confidence:.4f})")
        
        # STEP 3: Predict diet type
        print(f"\n[STEP 3] Predicting Diet Type...")
        diet_type = disease_to_diet_map.get(predicted_disease, "Balanced Diet")
        print(f"[STEP 3] Assigned Diet Type: '{diet_type}'")
        print(f"✅ Process Complete. Sending response to user.")
        print("="*50 + "\n")

        return {
            "extracted_symptoms": extracted_symptoms,
            "predicted_disease": predicted_disease,
            "disease_confidence": round(disease_confidence, 4),
            "recommended_diet_type": diet_type
        }

    except Exception as e:
        print(f"!!! CRITICAL ERROR in '/predict-diet-type': {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")






# --- API ENDPOINT 2: Generate Full Meal Plan ---
# @app.post("/generate-meal-plan")
# def generate_meal_plan_endpoint(input_data: MealPlanInput):
#     try:
#         print("\n" + "="*50)
#         print(f"🚀 New Request for Meal Plan Generation. Diet Type: '{input_data.diet_type}'")
#         print("="*50)
        
#         final_meal_plan = generate_final_diet_plan(input_data)
        
#         print(f"\n✅ Meal Plan generated successfully. Sending response to user.")
#         print("="*50 + "\n")
        
#         return final_meal_plan

#     except Exception as e:
#         print(f"!!! CRITICAL ERROR in '/generate-meal-plan': {e}")
#         raise HTTPException(status_code=500, detail="An internal server error occurred.")






# ---  API ENDPOINT 2: Generate Full Meal Plan ---
@app.post("/generate-meal-plan")
def generate_meal_plan_endpoint(input_data: MealPlanInput):
    try:
        print("\n" + "="*50)
        print(f"🚀 New Request for Meal Plan Generation. Diet Type: '{input_data.diet_type}'")
        print("="*50)
        
        # Call the new, fully implemented function
        final_meal_plan = generate_meal_plan_with_llm(input_data)
        
        # Check if the generation failed
        if "error" in final_meal_plan:
             raise HTTPException(status_code=500, detail=final_meal_plan)

        print(f"\n✅ Meal Plan generated successfully by LLM. Sending response to user.")
        print("="*50 + "\n")
        
        return final_meal_plan

    except Exception as e:
        print(f"!!! CRITICAL ERROR in '/generate-meal-plan': {e}")
        # Make sure we don't re-raise the HTTP exception we just made
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail="An internal server error occurred.")