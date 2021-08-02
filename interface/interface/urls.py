"""interface URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls import include, url
from django.contrib import admin
from django.urls import path
from django.views.generic import RedirectView
from input_output_preview.views import settings, imports, importFile, summary, home

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', RedirectView.as_view(url='/home/')),#Add URL maps to redirect the base URL to our application
    url(r'^home/$', home , name='home'),
    url(r'^settings/$', settings, name="settings"),
    url(r'^imports/$', imports, name="imports"),
    url(r'^importFile/$', importFile, name="importFile"),
    url(r'^summary/$', summary, name="summary"),
]
