import logging

from concept_search.taxonomy import Taxonomy
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm
from django.shortcuts import render, redirect

from .forms import NewUserForm
from .search_documents import search
from .search_wikipedia import search_wikipedia

logger = logging.getLogger("user_queries")
hdlr = logging.FileHandler("../../data/user_queries.log")
formatter = logging.Formatter("%(asctime)s %(message)s")
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)


# Create your views here.

# Taxonomy instantiation
tax = Taxonomy('../../data/external/acm_ccs.xml')


def index(request):
    """
    starting index page

    """
    if request.method == "GET":
        return render(
            request,
            "interfaces/home.html",
        )

def about(request):
    """
    about page

    """
    if request.method == "GET":
        return render(
            request,
            "interfaces/about.html",
        )

def search_results(request):
    if request.method == "GET":
        search_query = request.GET.get("search_query", None)
        logger.info(search_query)

        index_name = "papers"
        top_k = 15
        search_result = search(query=search_query, index=index_name, top_k=top_k)
        tax_query = tax.search_relationships(query=search_query)
        matched_wiki_page = search_wikipedia(query=search_query)

        context = {
            "search_result_list": search_result,
            "matched_wiki_page": matched_wiki_page,
            "unique_searches": len(search_result),
            "search_query": search_query,
            "concept_map": tax_query
        }

        return render(
            request=request,
            template_name="interfaces/search_result.html",
            context=context,
        )


def print_results(request, search_query):
    print(search_query)
    return render(request, "interfaces/home.html")


def register(request):
    if request.method == "POST":
        form = NewUserForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get("username")
            messages.success(request, f"New account created: {username}")
            login(request, user)
            return redirect("home")
        else:
            for msg in form.error_messages:
                messages.error(request, f"{msg}: {form.error_messages[msg]}")

            return render(
                request=request,
                template_name="users/register.html",
                context={"form": form},
            )

    form = NewUserForm
    return render(
        request=request, template_name="users/register.html", context={"form": form}
    )


def logout_request(request):
    logout(request)
    messages.info(request, "Logged out successfully!")
    return redirect("home")


def login_request(request):
    if request.method == "POST":
        form = AuthenticationForm(request=request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get("username")
            password = form.cleaned_data.get("password")
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.info(request, f"You are now logged in as {username}")
                return redirect("home")
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")
    form = AuthenticationForm()
    return render(
        request=request, template_name="users/login.html", context={"form": form}
    )
