# import google.generativeai as genai
# import asyncio


# genai.configure(api_key ="AIzaSyDkKn6vu2Qgx87jfsrevovbU8y7i0qdOy0")

# async def generate_question(skill):
#     """
#     Use the Gemini API to generate a technical interview question based on the given skill.
#     """
#     prompt = f"Generate a technical interview question which is simple for a candidate proficient in {skill}(dont provide output)."
#     model = genai.GenerativeModel("gemini-pro")
#     response = await asyncio.to_thread(model.generate_content, prompt)
#     return response.text.strip()

# async def evaluate_answer(candidate_answer, question):
#     """
#     Use the Gemini API to evaluate the candidate's answer to the given question.
#     """
#     prompt = f"Evaluate the following answer: {candidate_answer} to the question: {question}. Provide a score for answer out of 5 and explain."
#     model = genai.GenerativeModel("gemini-pro")
#     response = await asyncio.to_thread(model.generate_content, prompt)
#     return response.text.strip()

# async def start_interview():
#     """
#     Start the interview process by generating questions, capturing answers, and evaluating them using Gemini API.
#     """
#     candidate_name = "Rohith VK"
#     skills = ["Python", "Django", "Machine Learning", "Blockchain", "SQL"]

#     print(f"Starting interview with {candidate_name}...\n")

#     total_score = 0
#     num_questions = len(skills)  # One question per skill

#     for i, skill in enumerate(skills):
#         # Generate a new question using the Gemini API based on the skill
#         question = await generate_question(skill)
#         print(f"Question {i + 1}: {question}")

#         # Get the candidate's answer (simulated or actual input)
#         candidate_answer = input("Your answer: ")

#         # Evaluate the answer using the Gemini API
#         evaluation = await evaluate_answer(candidate_answer, question)

#         # Extract score from the evaluation (assuming Gemini provides a score like "Score: X out of 5")
#         try:
#             score = int(evaluation.split("Score: ")[1].split(" out of 5")[0])
#         except (IndexError, ValueError):
#             score = 0  # If there's an error parsing the score, default to 0

#         total_score += score

#         # Output the evaluation result
#         print(f"Evaluation: {evaluation}\n")

#     # Calculate and print final score
#     # print(f"Interview completed. Final score: {total_score} out of {num_questions * 5}.")

# asyncio.run(start_interview())


import google.generativeai as genai
import asyncio

genai.configure(api_key="AIzaSyDkKn6vu2Qgx87jfsrevovbU8y7i0qdOy0")

async def generate_question(skill):
    """
    Use the Gemini API to generate a technical interview question based on the given skill.
    """
    prompt = f"Generate a technical interview question which is simple for a candidate proficient in {skill}(don't provide output)."
    model = genai.GenerativeModel("gemini-pro")
    response = await asyncio.to_thread(model.generate_content, prompt)
    return response.text.strip()

async def evaluate_answer(candidate_answer, question):
    """
    Use the Gemini API to evaluate the candidate's answer to the given question.
    """
    prompt = f"Evaluate the following answer: {candidate_answer} to the question: {question}. Provide a score for answer out of 5 and explain."
    model = genai.GenerativeModel("gemini-pro")
    response = await asyncio.to_thread(model.generate_content, prompt)
    return response.text.strip()

async def start_interview():
    """
    Start the interview process by generating questions, capturing answers, and evaluating them using Gemini API.
    """
    candidate_name = input("Enter candidate name: ")
    skills_input = input("Enter the skills for the interview, separated by commas: ")
    skills = [skill.strip() for skill in skills_input.split(",")]

    print(f"\nStarting interview with {candidate_name}...\n")

    total_score = 0
    num_questions = len(skills)  # One question per skill

    for i, skill in enumerate(skills):
        # Generate a new question using the Gemini API based on the skill
        question = await generate_question(skill)
        print(f"Question {i + 1}: {question}")

        # Get the candidate's answer (simulated or actual input)
        candidate_answer = input("Your answer: ")

        # Evaluate the answer using the Gemini API
        evaluation = await evaluate_answer(candidate_answer, question)

        # Extract score from the evaluation (assuming Gemini provides a score like "Score: X out of 5")
        try:
            score = int(evaluation.split("Score: ")[1].split(" out of 5")[0])
        except (IndexError, ValueError):
            score = 0  # If there's an error parsing the score, default to 0

        total_score += score

        # Output the evaluation result
        print(f"Evaluation: {evaluation}\n")

    # Calculate and print final score
    print(f"Interview completed. Final score: {total_score} out of {num_questions * 5}.")

asyncio.run(start_interview())

