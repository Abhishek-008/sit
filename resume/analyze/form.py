from django import forms


class ResumeUploadForm(forms.Form):
    file = forms.FileField(label='Upload your resume')


from django import forms

from django import forms

class UploadForm(forms.Form):
    resumes = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple':False}), required=True)
    job_description = forms.FileField(required=True)

