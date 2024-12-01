from transformers import pipeline
import os

def load_model():
    """Load the open-source language model."""
    model_name = "EleutherAI/gpt-neo-1.3B"  # Adjust model size based on your hardware
    generator = pipeline('text-generation', model=model_name)
    return generator
# EleutherAI/gpt-j-6B

def analyze_usage(generator, prompt):
    """Use the language model to analyze energy usage."""
    response = generator(
        prompt,
        max_new_tokens=200,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        return_full_text=False
    )
    generated_text = response[0]['generated_text'].strip()
    return generated_text



def get_appliance_suggestion(generator, timestamp, user_data):
    """Generate appliance suggestion based on energy spike."""
    prompt = (f"At {timestamp}, there was a spike in energy consumption. "
              f"Based on the user's habits: {user_data}, which appliance is likely in use and why?")
    suggestion = analyze_usage(generator, prompt)
    return suggestion

def get_anomaly_explanation(generator, timestamp, user_data):
    """Provide explanation for an anomaly."""
    prompt = (
        f"An unusually high energy consumption was detected at {timestamp}.\n"
        f"Given the user's habits: {user_data}, what could be the possible reasons?\n"
        "Please provide a brief explanation."
    )
    explanation = analyze_usage(generator, prompt)
    return explanation


def get_energy_saving_tips(generator, consumption_data, user_data):
    """Provide personalized energy-saving tips."""
    prompt = (
        f"Based on the user's energy consumption data:\n{consumption_data}\n"
        f"and their habits: {user_data}, provide a list of personalized tips to improve energy efficiency.\n"
        "Please present the tips in bullet points."
    )
    tips = analyze_usage(generator, prompt)
    return tips


if __name__ == "__main__":
    # Load the model
    generator = load_model()
    
    # Example usage
    timestamp = "2024-08-07 18:00"
    user_data = "The user usually cooks dinner between 6 PM and 8 PM, uses air conditioning during hot days, and works from home."
    consumption_data = "High energy usage during afternoon hours, moderate usage at night."
    
    appliance_suggestion = get_appliance_suggestion(generator, timestamp, user_data)
    print(f"Appliance Suggestion: {appliance_suggestion}")
    
    anomaly_explanation = get_anomaly_explanation(generator, timestamp, user_data)
    print(f"Anomaly Explanation: {anomaly_explanation}")
    
    energy_saving_tips = get_energy_saving_tips(generator, consumption_data, user_data)
    print(f"Energy Saving Tips: {energy_saving_tips}")
