
from django.conf import settings
from django.shortcuts import render, HttpResponse
from .form import ResumeUploadForm
import os
import pickle
import docx
import PyPDF2
import re



def index2(request):
    return render(request,'home2.html')

def index1(request):
    return render(request,'home1.html')

def index(request):
    return render(request,'home.html')
# Load pre-trained model, TF-IDF vectorizer, and label encoder from the static directory
model_path = os.path.join(settings.STATICFILES_DIRS[0], 'models')

svc_model = pickle.load(open(os.path.join(model_path, 'clf.pkl'), 'rb'))
tfidf = pickle.load(open(os.path.join(model_path, 'tfidf.pkl'), 'rb'))
le = pickle.load(open(os.path.join(model_path, 'encoder.pkl'), 'rb'))

# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

# Function to extract text from TXT
def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text

# Function to handle file upload and text extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text

# Function to predict the category of a resume
def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = svc_model.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)
    return predicted_category_name[0]

# View for handling resume uploads and predictions
def upload_resume(request):
    if request.method == 'POST':
        form = ResumeUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            uploaded_file = request.FILES.get('file')
            
            if uploaded_file:
                try:
                    # Handle file and extract text
                    resume_text = handle_file_upload(uploaded_file)
                    
                    # Make prediction
                    predicted_category = pred(resume_text)

                    # Render response with predicted category
                    return render(request, 'upload_resume.html', {
                        'form': form,
                        'predicted_category': predicted_category,
                        'resume_text': resume_text,  # Optionally display the extracted text
                    })
                except Exception as e:
                    return render(request, 'upload_resume.html', {
                        'form': form,
                        'error': f"Error during processing: {str(e)}",
                    })
            else:
                # File not uploaded or processed correctly
                return render(request, 'upload_resume.html', {
                    'form': form,
                    'error': "No file uploaded. Please upload a file.",
                })
    else:
        form = ResumeUploadForm()

    return render(request, 'upload_resume.html', {'form': form})




from django.shortcuts import render
from .processor import process_resume
import os

import os
from django.conf import settings


def upload_view(request):
    context = {"results": None}  # Initialize context with no results

    if request.method == "POST":
        resumes = request.FILES.getlist("resumes")
        jd_file = request.FILES["jd"]
        skills_file = request.FILES["skills"]

        # Read job description and skills file
        jd_text = jd_file.read().decode("utf-8")
        skills_set = set(skills_file.read().decode("utf-8").splitlines())

        results = []
        for resume in resumes:
            resume_path = f"temp_{resume.name}"
            with open(resume_path, "wb") as temp_file:
                temp_file.write(resume.read())
            
            # Process the resume
            result = process_resume(resume_path, jd_text, skills_set)
            results.append(result)

            # Clean up temporary file
            os.remove(resume_path)

        # Add results to the context
        context["results"] = results

    return render(request, "upload_and_results.html",context)



from django.http import JsonResponse
from django.shortcuts import render
import google.generativeai as genai
import asyncio

# Configure the API key
genai.configure(api_key="AIzaSyDkKn6vu2Qgx87jfsrevovbU8y7i0qdOy0")

# Function to generate interview questions
async def generate_question(skill):
    prompt = f"Generate a technical interview question which is simple for a candidate proficient in {skill}(don't provide output)."
    model = genai.GenerativeModel("gemini-pro")
    response = await asyncio.to_thread(model.generate_content, prompt)
    return response.text.strip()

# Function to evaluate the candidate's answer
async def evaluate_answer(candidate_answer, question):
    prompt = f"Evaluate the following answer: {candidate_answer} to the question: {question}. Provide a score for answer given from the candidate out of 5 and explain."
    model = genai.GenerativeModel("gemini-pro")
    response = await asyncio.to_thread(model.generate_content, prompt)
    return response.text.strip()

# Main view to start the interview process
def interview_view(request):
    if request.method == 'POST':
        candidate_name = request.POST.get('candidate_name')
        skills_input = request.POST.get('skills')
        skills = [skill.strip() for skill in skills_input.split(",")]

        # Start the interview asynchronously and get questions
        questions_and_evaluations = asyncio.run(start_interview(candidate_name, skills))

        return JsonResponse(questions_and_evaluations, safe=False)

    return render(request, 'interview.html')

# Async function to start the interview, generate questions, capture answers, and evaluate them
async def start_interview(candidate_name, skills):
    total_score = 0
    num_questions = len(skills)
    questions_and_evaluations = []

    for i, skill in enumerate(skills):
        question = await generate_question(skill)

        # Simulating user input for answers here (for real use, replace with dynamic input in views)
        candidate_answer = "Simulated answer for interview"  # This should be replaced by actual input handling

        evaluation = await evaluate_answer(candidate_answer, question)

        try:
            score = int(evaluation.split("Score: ")[1].split(" out of 5")[0])
        except (IndexError, ValueError):
            score = 0  # Default score in case of error

        total_score += score
        questions_and_evaluations.append({
            "question": question,
            "evaluation": evaluation,
            "score": score
        })

    return {"questions": questions_and_evaluations, "total_score": total_score, "max_score": num_questions * 5}

# View to evaluate answers and get scores asynchronously
async def evaluate_answer_view(request):
    if request.method == 'POST':
        candidate_answer = request.POST.get('answer')
        question = request.POST.get('question')

        # Call the evaluation function
        evaluation = await evaluate_answer(candidate_answer, question)

        # Extract score from the evaluation text (assuming "Score: X out of 5")
        try:
            score = int(evaluation.split("Score: ")[1].split(" out of 5")[0])
        except (IndexError, ValueError):
            score = 0  # Default to 0 if score cannot be parsed

        return JsonResponse({
            'score': score,
            'evaluation': evaluation
        })

async def evaluate_answer(candidate_answer, question):
    prompt = f"Evaluate the following answer: {candidate_answer} to the question: {question}. Provide a score for answer out of 5 and explain."
    model = genai.GenerativeModel("gemini-pro")
    response = await asyncio.to_thread(model.generate_content, prompt)
    return response.text.strip()





import os
from django.shortcuts import render
from django.conf import settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.core.files.storage import default_storage
import docx2txt
import PyPDF2

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        return ""

def match_resume(request):
    if request.method == 'POST':
        # Get the job description file
        job_description_file = request.FILES.get('job_description_file')
        if job_description_file:
            # Save and read the job description file
            job_description_filename = default_storage.save(os.path.join('uploads', job_description_file.name), job_description_file)
            job_description_path = os.path.join(settings.MEDIA_ROOT, job_description_filename)
            job_description = extract_text_from_txt(job_description_path)
        else:
            return render(request, 'matchresume.html', {'message': "Please upload a job description file."})

        # Get the resumes
        resume_files = request.FILES.getlist('resumes')
        resumes = []
        filenames = []
        for resume_file in resume_files:
            filename = default_storage.save(os.path.join('uploads', resume_file.name), resume_file)
            file_path = os.path.join(settings.MEDIA_ROOT, filename)
            filenames.append(resume_file.name)
            resumes.append(extract_text(file_path))

        if not resumes:
            return render(request, 'matchresume.html', {'message': "Please upload resumes."})

        # Vectorize job description and resumes
        vectorizer = TfidfVectorizer().fit_transform([job_description] + resumes)
        vectors = vectorizer.toarray()

        # Calculate cosine similarities
        job_vector = vectors[0]
        resume_vectors = vectors[1:]
        similarities = cosine_similarity([job_vector], resume_vectors)[0]

        # Get top 5 resumes and their similarity scores
        top_indices = similarities.argsort()[-5:][::-1]
        top_resumes = [filenames[i] for i in top_indices]
        similarity_scores = [round(similarities[i], 2) for i in top_indices]

        return render(request, 'matchresume.html', {
            'message': "Top matching resumes:",
            'top_resumes': zip(top_resumes, similarity_scores)
        })

    return render(request, 'matchresume.html')
