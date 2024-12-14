from django.contrib import admin
from django.urls import path
from analyze import views
from resume import settings
from django.urls import path
from . import views
from django.conf.urls.static import static

urlpatterns = [
    path('', views.index, name='home'),
    path('home1/', views.index1, name='home1'),
    path('home2/', views.index2, name='home2'),
    path('upload/', views.upload_resume, name='upload_resume'),
    path('updown/', views.upload_view, name='upload_view'),
    path('interview/', views.interview_view, name='interview'),
    path('evaluate_answer/', views.evaluate_answer_view, name='evaluate_answer'),
    path('match_resume/', views.match_resume, name='match_resume'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
