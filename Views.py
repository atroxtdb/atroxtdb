import datetime
import calendar
import csv
import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth, HTTPProxyAuth
from datetime import timedelta
from django_htmx.http import trigger_client_event
from django.shortcuts import render, reverse
from django.views.generic import CreateView, UpdateView, ListView, DeleteView, DetailView, edit
from django.db.models import Count, Sum, Avg, CharField, Case, Value, When, F, Q, Max, Min
from django.db.models.functions import ExtractWeek, ExtractYear, ExtractMonth
from django.contrib.auth.mixins import LoginRequiredMixin, AccessMixin
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView, LogoutView
from django.http import HttpResponseRedirect
from .models import *
from .forms import *


@login_required(login_url="/homelogin/")
def redirect_home(request):
    return HttpResponseRedirect(reverse('workcalender'))


class CreateProject(LoginRequiredMixin, CreateView):
    template_name = "projectdata/projectdata_create.html"
    form_class = ProjectModelForm
    login_url = '/homelogin/'
    redirect_field_name = 'redirect_to'
    success_message = "%(projectdata)s was created successfully"

    def get_success_url(self):
        return reverse("Home")

    def form_valid(self, form):
        projectdata = form.save(commit=False)
        projectdata.created_by = self.request.user
        print(self.request.user.username + " Create Project")
        query = Project.objects.filter(project_code=projectdata.project_code,
                                       project_stage=projectdata.project_stage,
                                       spec=projectdata.spec)
        if len(query) > 0:
            content = "{}_{} with  already exists".format(projectdata.project_code,
                                                          projectdata.project_stage,
                                                          projectdata.spec)
            messages.error(request=self.request, message=content)
            return HttpResponseRedirect(self.get_success_url())

        projectdata.dueDate = form.cleaned_data.get('dueDate')
        projectdata.receivedDate = form.cleaned_data.get('receivedDate')
        projectdata.save()
        return super(CreateProject, self).form_valid(form)


class UpdateProject(LoginRequiredMixin, UpdateView):
    template_name = "projectdata/projectdata_edit.html"
    form_class = ProjectModelForm
    login_url = '/homelogin/'
    redirect_field_name = 'redirect_to'
    success_message = "%(projectdata)s was updated successfully"
    context_object_name = "todolist"

    def get_queryset(self):
        queryset = Project.objects.filter(pk=self.kwargs['pk'])
        return queryset

    def get_success_url(self):
        return reverse("Home")

    # def get_context_data(self, **kwargs):
    #     context = super(UpdateProject, self).get_context_data(**kwargs)
    #     context['pk'] = self.kwargs['pk']
    #     return context
    def form_valid(self, form):
        print(self.request.user.username + " Updated Project")

        projectdata = form.save(commit=False)
        projectdata.receivedDate = form.cleaned_data.get('receivedDate')
        projectdata.dueDate = form.cleaned_data.get('dueDate')
        query = Project.objects.filter(project_code=projectdata.project_code,
                                       project_stage=projectdata.project_stage)
        if len(query) > 0:
            if query.values('pk')[0]['pk'] == self.kwargs['pk']:
                projectdata.save()
            else:
                content = "{}_{} with  already exists".format(projectdata.project_code,
                                                              projectdata.project_stage, )
                messages.error(request=self.request, message=content)
            return HttpResponseRedirect(self.get_success_url())
        projectdata.save()
        return super(UpdateProject, self).form_valid(form)


class ViewProjects(LoginRequiredMixin, ListView):
    template_name = "projectdata/projectdata_list.html"
    context_object_name = "projects"
    login_url = '/homelogin/'
    redirect_field_name = 'redirect_to'

    def get_queryset(self):
        if 'month' in self.kwargs:
            year = self.kwargs['year']
            if self.kwargs['month'] != 0:
                queryset = Project.objects.distinct().order_by('project_code')
                if year != 0:
                    queryset = queryset.filter(receivedDate__year=year)
                queryset = queryset.filter(
                    Q(dueDate__month=self.kwargs['month']) | Q(receivedDate__month=self.kwargs['month'])).order_by(
                    '-dueDate')
            else:
                if year != 0:
                    queryset = Project.objects.filter(receivedDate__year=year).distinct().order_by(
                        'project_code').order_by('-dueDate')
                else:
                    queryset = Project.objects.all()
        else:
            queryset = Project.objects.distinct().order_by('project_code')
            queryset = queryset.filter(receivedDate__year=datetime.date.today().year)
            queryset = queryset.filter(Q(dueDate__month=datetime.date.today().month) | Q(
                receivedDate__month=datetime.date.today().month)).order_by('-dueDate')
        return queryset

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super(ViewProjects, self).get_context_data(**kwargs)
        if 'month' in self.kwargs:
            context['month'] = self.kwargs['month']
            context['year'] = self.kwargs['year']
        else:
            context['month'] = datetime.date.today().month
            context['year'] = datetime.date.today().year
        for query in self.object_list:
            if self.request.user.hod:
                open_projects = ListOfItems.objects.filter(project=query, status="On going")
            elif self.request.user.hos:
                open_projects = ListOfItems.objects.filter(project=query, status="On going",
                                                           created_by__section=self.request.user.section)
            else:
                open_projects = ListOfItems.objects.filter(project=query, status="On going",
                                                           system=self.request.user.system)
            if open_projects.count() >= 1:
                if 'open_project' not in context:
                    context['open_project'] = []
                context['open_project'].append(query)
        print(self.request.user.username + " Viewed Project")
        return context


class DeleteProject(LoginRequiredMixin, DeleteView):
    template_name = "projectdata/projectdata_delete.html"
    context_object_name = "todolist"
    login_url = '/homelogin/'
    redirect_field_name = 'redirect_to'

    def get_success_url(self):
        return reverse('Home')

    def get_queryset(self):
        print(self.request.user.username + " Deleted Project")
        queryset = Project.objects.filter(pk=self.kwargs['pk'])
        return queryset

    # def get_context_data(self, **kwargs):
    #     context = super(DeleteProject, self).get_context_data(**kwargs)
    #     context['pk'] = self.kwargs['pk']
    #     return context


@login_required(login_url="/homelogin/")
def createWork(request):
    initial = {'system': request.user.system,
               'startDate': datetime.date.today(),
               'endDate': datetime.date.today() + datetime.timedelta(days=1)}
    form = WorkModelForm(initial=initial)
    context = {"form": form}
    return render(request, "partials/work_create_form.html", context)


@login_required(login_url="/homelogin/")
def detail_work(request, pk):
    work = ListOfItems.objects.get(pk=pk)
    context = {"project": work}
    return render(request, "partials/work_detail_form.html", context)


class CreateWork(LoginRequiredMixin, CreateView):
    template_name = "projectdata/work_create.html"
    form_class = WorkModelForm
    login_url = '/homelogin/'
    redirect_field_name = 'redirect_to'
    context_object_name = "projects"
    redirect_pk = 0

    def get_initial(self):
        base_initial = super(CreateWork, self).get_initial()
        base_initial['startDate'] = datetime.date.today()
        base_initial['endDate'] = Project.objects.get(pk=self.kwargs['pk']).dueDate
        if Project.objects.get(pk=self.kwargs['pk']).project_code == "Education":
            base_initial['subsystem'] = SubSystem.objects.get(system=self.request.user.system, typeOfWork="Education")
        return base_initial

    def get_success_url(self):
        if self.redirect_pk == 0:
            return reverse('Home')
        else:
            return reverse("detail-work", kwargs={'pk': self.redirect_pk})

    def get_queryset(self):
        if self.request.user.hod:
            queryset = ListOfItems.objects.filter(project_id=self.kwargs['pk']).order_by('subsystem', 'assigned_to',
                                                                                         'startDate')
        elif self.request.user.hos:
            queryset = ListOfItems.objects.filter(project_id=self.kwargs['pk'],
                                                  created_by__section=self.request.user.section,
                                                  ).order_by('subsystem', 'assigned_to', 'startDate')
        else:
            queryset = ListOfItems.objects.filter(project_id=self.kwargs['pk'],
                                                  system=self.request.user.system).order_by('subsystem', 'assigned_to',
                                                                                            'startDate')
        # print(queryset.order_by().values('system').distinct())
        return queryset

    def get_context_data(self, **kwargs):
        context = super(CreateWork, self).get_context_data(**kwargs)
        context['projects'] = self.get_queryset()
        context['project_code'] = self.kwargs['pk']
        context['project_detail'] = Project.objects.get(pk=self.kwargs['pk'])
        context['project_summary'] = self.get_queryset().values('system__name').order_by().annotate(
            total_count=Sum('issueCount')).annotate(on_going_status=Count('status', filter=Q(status="On going")))
        return context

    def form_valid(self, form):
        projectdata = form.save(commit=False)
        projectdata.project_id = self.kwargs['pk']
        projectdata.typeOfWork = projectdata.subsystem.typeOfWork
        projectdata.startDate = form.cleaned_data.get('startDate')
        projectdata.endDate = form.cleaned_data.get('endDate')
        projectdata.system = form.cleaned_data.get('system')
        projectdata.subsystem = form.cleaned_data.get('subsystem')
        projectdata.created_by = self.request.user
        print(self.request.user.username + " Created Work")
        if projectdata.typeOfWork == "Education":
            newtask = ListOfItems.objects.create(project_id=self.kwargs['pk'],
                                                 system=projectdata.system,
                                                 subsystem=projectdata.subsystem,
                                                 typeOfWork=projectdata.typeOfWork,
                                                 startDate=form.cleaned_data.get('startDate'),
                                                 endDate=form.cleaned_data.get('endDate'),
                                                 created_by=self.request.user,
                                                 issueCount=form.cleaned_data.get('issueCount'),
                                                 status=form.cleaned_data.get('status'),
                                                 remarks=form.cleaned_data.get('remarks'),
                                                 )
            newtask.assigned_to.add(self.request.user)
            newtask.excludeDates.set(form.cleaned_data.get('excludeDates'))
            self.redirect_pk = newtask.pk
            newtask.save()
            return HttpResponseRedirect(self.get_success_url())
        else:
            if form.cleaned_data.get('assigned_to').count() == 0:
                query = ListOfItems.objects.filter(project_id=self.kwargs['pk'],
                                                   system=projectdata.system,
                                                   subsystem=projectdata.subsystem,
                                                   assigned_to=self.request.user)
                if len(query) == 0:
                    newtask = ListOfItems.objects.create(project_id=self.kwargs['pk'],
                                                         system=projectdata.system,
                                                         subsystem=projectdata.subsystem,
                                                         typeOfWork=projectdata.typeOfWork,
                                                         startDate=form.cleaned_data.get('startDate'),
                                                         endDate=form.cleaned_data.get('endDate'),
                                                         created_by=self.request.user,
                                                         issueCount=form.cleaned_data.get('issueCount'),
                                                         status=form.cleaned_data.get('status'),
                                                         remarks=form.cleaned_data.get('remarks'),
                                                         )
                    newtask.assigned_to.add(self.request.user)
                    newtask.excludeDates.set(form.cleaned_data.get('excludeDates'))
                    newtask.save()
                    self.redirect_pk = newtask.pk
                    return HttpResponseRedirect(self.get_success_url())
                else:
                    content = "{}_{} with {} already exists in {}'s name".format(projectdata.project.project_code,
                                                                                 projectdata.project.project_stage,
                                                                                 projectdata.subsystem,
                                                                                 self.request.user)
                    messages.error(request=self.request, message=content)
            elif form.cleaned_data.get('assigned_to').count() > 1:
                for user in form.cleaned_data.get('assigned_to'):
                    query = ListOfItems.objects.filter(project_id=self.kwargs['pk'],
                                                       system=projectdata.system,
                                                       subsystem=projectdata.subsystem,
                                                       assigned_to=user)
                    if len(query) > 0:
                        content = "{}_{} with {} already exists in {}'s name".format(projectdata.project.project_code,
                                                                                     projectdata.project.project_stage,
                                                                                     projectdata.subsystem,
                                                                                     user)
                        messages.error(request=self.request, message=content)
                    else:
                        newtask = ListOfItems.objects.create(project_id=self.kwargs['pk'],
                                                             system=projectdata.system,
                                                             subsystem=projectdata.subsystem,
                                                             typeOfWork=projectdata.typeOfWork,
                                                             startDate=form.cleaned_data.get('startDate'),
                                                             endDate=form.cleaned_data.get('endDate'),
                                                             created_by=self.request.user,
                                                             issueCount=form.cleaned_data.get('issueCount'),
                                                             status=form.cleaned_data.get('status'),
                                                             remarks=form.cleaned_data.get('remarks'),
                                                             )
                        newtask.excludeDates.set(form.cleaned_data.get('excludeDates'))
                        newtask.assigned_to.add(user)
                        newtask.save()
                        self.redirect_pk = newtask.pk
                return HttpResponseRedirect(self.get_success_url())
            else:
                query = ListOfItems.objects.filter(project_id=self.kwargs['pk'],
                                                   system=projectdata.system,
                                                   subsystem=projectdata.subsystem,
                                                   assigned_to=form.cleaned_data.get('assigned_to')[0])
                if len(query) > 0:
                    project = Project.objects.get(pk=self.kwargs['pk'])
                    content = "{}_{} with {} already exists in {}'s name".format(project.project_code,
                                                                                 project.project_stage,
                                                                                 projectdata.subsystem,
                                                                                 form.cleaned_data.get('assigned_to')[
                                                                                     0])
                    messages.error(request=self.request, message=content)
                    return HttpResponseRedirect(self.get_success_url())
                else:
                    projectdata.save()
                    projectdata.excludeDates.set(form.cleaned_data.get('excludeDates'))
                    self.redirect_pk = projectdata.pk
        return super(CreateWork, self).form_valid(form)


class UpdateWork(LoginRequiredMixin, UpdateView):
    template_name = "partials/work_create_form.html"
    form_class = WorkModelForm
    login_url = '/homelogin/'
    context_object_name = 'projects'
    redirect_field_name = 'redirect_to'

    def get_queryset(self):
        queryset = ListOfItems.objects.filter(pk=self.kwargs['pk'])
        return queryset

    def get_initial(self):
        base_initial = super(UpdateWork, self).get_initial()
        try:
            if self.get_object().assigned_to.all():
                base_initial['id'] = self.get_object().pk
                base_initial['system'] = self.get_object().system
                base_initial['subsystem'] = self.get_object().subsystem
                if self.get_object().status == 'On going':
                    base_initial['endDate'] = datetime.date.today()
                    base_initial['status'] = "Finished"
            else:
                base_initial['id'] = self.get_object().pk
                base_initial['startDate'] = datetime.date.today()
                base_initial['system'] = self.get_object().system
                base_initial['subsystem'] = self.get_object().subsystem
                base_initial['status'] = "On going"
        except:
            pass
        return base_initial

    def get_success_url(self):
        return reverse("detail-work", kwargs={'pk': self.get_object().pk})

    def get_context_data(self, **kwargs):
        context = super(UpdateWork, self).get_context_data(**kwargs)
        context['id'] = self.get_object().id
        return context

    def form_valid(self, form):
        projectdata = form.save(commit=False)
        projectdata.system = form.cleaned_data.get('system')
        projectdata.subsystem = form.cleaned_data.get('subsystem')
        projectdata.typeOfWork = projectdata.subsystem.typeOfWork
        projectdata.startDate = form.cleaned_data.get('startDate')
        projectdata.endDate = form.cleaned_data.get('endDate')
        projectdata.excludeDates.set(form.cleaned_data.get('excludeDates'))
        projectdata.created_by = self.request.user
        print(self.request.user.username + " Updated Work")
        if projectdata.typeOfWork == "Education":
            projectdata.save()
            return HttpResponseRedirect(self.get_success_url())
        if form.cleaned_data.get('assigned_to').count() > 1:
            newcount = 0
            for user in form.cleaned_data.get('assigned_to'):
                query = ListOfItems.objects.filter(project_id=self.get_object().project.pk)
                query = query.filter(subsystem=projectdata.subsystem)
                query = query.filter(assigned_to=user)
                if len(query) > 0:
                    if query.values('pk')[0]['pk'] == self.kwargs['pk']:
                        projectdata.save()
                    else:
                        content = "{}_{} with {} already exists in {}'s name".format(projectdata.project.project_code,
                                                                                     projectdata.project.project_stage,
                                                                                     projectdata.subsystem,
                                                                                     user)
                        messages.error(request=self.request, message=content)
                else:
                    newcount += 1
                    newtask = ListOfItems.objects.create(project_id=self.get_object().project.pk,
                                                         system=projectdata.system,
                                                         subsystem=projectdata.subsystem,
                                                         typeOfWork=projectdata.typeOfWork,
                                                         startDate=form.cleaned_data.get('startDate'),
                                                         endDate=form.cleaned_data.get('endDate'),
                                                         created_by=self.request.user,
                                                         issueCount=form.cleaned_data.get('issueCount'),
                                                         remarks=form.cleaned_data.get('remarks'),
                                                         )
                    newtask.excludeDates.set(form.cleaned_data.get('excludeDates'))
                    newtask.assigned_to.add(user)
            if newcount == form.cleaned_data.get('assigned_to').count():
                ListOfItems.objects.get(pk=self.kwargs['pk']).delete()
            return HttpResponseRedirect(self.get_success_url())
        else:
            try:
                assigned_to = form.cleaned_data.get('assigned_to')[0]
            except:
                assigned_to = self.request.user
                projectdata.assigned_to.add(assigned_to)
            query = ListOfItems.objects.filter(project_id=self.get_object().project.pk)
            query = query.filter(subsystem=projectdata.subsystem)
            query = query.filter(assigned_to=assigned_to)

            if len(query) > 0:
                if query.values('pk')[0]['pk'] == self.kwargs['pk']:
                    projectdata.save()
                    return HttpResponseRedirect(self.get_success_url())
                else:
                    content = "{}_{} with {} already exists in {}'s name".format(projectdata.project.project_code,
                                                                                 projectdata.project.project_stage,
                                                                                 projectdata.subsystem,
                                                                                 form.cleaned_data.get('assigned_to')[
                                                                                     0])
                    messages.error(request=self.request, message=content)
                    return HttpResponseRedirect(self.get_success_url())
            else:
                projectdata.save()
        return super(UpdateWork, self).form_valid(form)


class DeleteWork(LoginRequiredMixin, DeleteView):
    template_name = "projectdata/work_delete.html"
    context_object_name = "todolist"
    login_url = '/homelogin/'
    redirect_field_name = 'redirect_to'

    def get_success_url(self):
        return ""

    def get_queryset(self):
        print(self.request.user.username + " Deleted Work")
        queryset = ListOfItems.objects.filter(pk=self.kwargs['pk'], system=self.request.user.system)
        return queryset


@login_required(login_url="/homelogin/")
def load_subsystems(request):
    system_id = request.GET.get('system')
    if request.GET.get('subsystem'):
        subsys = request.GET.get('subsystem')
    else:
        subsys = SubSystem.objects.get(system=system_id, typeOfWork="Education").pk
    subsystems = SubSystem.objects.filter(system_id=system_id).order_by('name')
    return render(request, 'projectdata/subsystem_choices.html', {'subsystems': subsystems, 'subsys': int(subsys)})


@login_required(login_url="/homelogin/")
def filter_user(request):
    if request.GET.get('assigned_to'):
        assigned_to = request.GET.get('assigned_to')
    else:
        assigned_to = 0
    system_id = request.GET.get('system')
    users = User.objects.filter(system_id=system_id).order_by('username').exclude(hos=True).exclude(hod=True)
    users |= User.objects.filter(pk=request.user.pk)
    return render(request, 'projectdata/user_filter.html', {'users': users, 'sub': int(assigned_to)})


@login_required(login_url="/homelogin/")
def filter_center(request):
    if request.GET.get('project_code'):
        project_code = request.GET.get('project_code')
        center = Project.objects.filter(project_code=project_code).values('center')
        if request.GET.get('project_stage'):
            project_stage = request.GET.get('project_stage')
            if len(Project.objects.filter(project_code=project_code,project_stage=project_stage)) > 0:
                center = Project.objects.filter(project_code=project_code, project_stage=project_stage).values('center')
    centers = {'COMPACT','CHINA','COACH BUS','GENESIS','HEAVY TRUCK','HMIE S','LIGHT BUS','LIGHT COMM VEHICLE',
               'LIGHT TRUCK','MEDIUM TRUCK','MID','RV','SEMI BONNET','TRANSIT BUS'}
    return render(request, 'projectdata/center_filter.html', {'centers': centers, 'sub': center[0]})


@login_required(login_url="/homelogin/")
def filter_company(request):
    if request.GET.get('project_code'):
        project_code = request.GET.get('project_code')
        company = Project.objects.filter(project_code=project_code).values('company')
    companies = {'HMC','KMC','HMIE'}
    return render(request, 'projectdata/company_filter.html', {'centers': companies, 'sub': company[0]})



@login_required(login_url="/homelogin/")
def filter_dates(request):
    if request.GET.get('startDate'):
        startDate = request.GET.get('startDate')
    else:
        startDate = datetime.date.today()
    if request.GET.get('endDate'):
        endDate = request.GET.get('endDate')
    else:
        endDate = datetime.date.today()
    dates = Dates.objects.filter(date__gte=startDate, date__lte=endDate)
    if request.GET.getlist('excludeDates'):
        sub = request.GET.getlist('excludeDates')
        sub = [int(i) for i in sub]
    else:
        sub = []
        holiday_query = dates.filter(holiday=True)
        if len(holiday_query) > 0:
            for item in holiday_query:
                sub.append(item.pk)
    return render(request, 'projectdata/date_filter.html', {'dates': dates, 'sub': sub})


@login_required(login_url="/homelogin/")
def breakfast_check(request):
    if request.GET.get('intime'):
        intime = request.GET.get('intime')
        try:
            intime = datetime.datetime.strptime(intime, '%H:%M').time()
        except:
            intime = datetime.datetime.strptime(intime, '%H:%M:%S').time()
        if intime < datetime.time(hour=7, minute=45):
            breakfast = True
        else:
            breakfast = False
    else:
        breakfast = True
    return render(request, 'projectdata/checkbox.html', {'breakfast': breakfast})


class UserLogin(LoginView):
    template_name = "projectdata/user_login.html"

    def get_success_url(self):
        return reverse("Home")

    def get_redirect_url(self):
        return reverse("Home")


class UserLogout(LogoutView):
    template_name = "projectdata/user_logout.html"

    @staticmethod
    def get_success_url():
        return reverse("login")


class SignupView(CreateView):
    template_name = "projectdata/user_signup.html"
    form_class = CustomUserCreationForm

    def get_success_url(self):
        return reverse("login")

    def get_context_data(self, **kwargs):
        context = super(SignupView, self).get_context_data(**kwargs)
        return context


class ViewAssignement(LoginRequiredMixin, ListView):
    template_name = "projectdata/work_assignement.html"
    context_object_name = "projects"
    login_url = "/homelogin/"
    redirect_field_name = "redirect_to"

    def get_queryset(self):
        queryset = ListOfItems.objects.filter(
            Q(assigned_to=self.request.user, status='On going', endDate__month__gte=datetime.date.today().month)
            | Q(assigned_to=None, status='On going', system=self.request.user.system,
                endDate__month__gte=datetime.date.today().month))
        queryset = queryset.filter(endDate__year__gte=datetime.date.today().year)
        print(self.request.user.username + " Viewed Pending Tasks")
        return queryset


class MandayView(LoginRequiredMixin, ListView):
    template_name = "projectdata/manday_list.html"
    context_object_name = "todolists"
    login_url = '/homelogin/'
    redirect_field_name = 'redirect_to'

    def get_queryset(self):
        if 'month' in self.kwargs:
            month = self.kwargs['month']
            year = self.kwargs['year']
        else:
            month = datetime.date.today().month
            year = datetime.date.today().year
        if month == 0 and year == 0:
            queryset = Man_Hours_model.objects.filter(user=self.request.user)
        elif month == 0:
            queryset = Man_Hours_model.objects.filter(user=self.request.user, year=year)
        elif year == 0:
            queryset = Man_Hours_model.objects.filter(user=self.request.user, month=month)
        else:
            queryset = Man_Hours_model.objects.filter(month=month, user=self.request.user, year=year)
        queryset = queryset.order_by('user__username', 'date')
        print(self.request.user.username + " Viewed Manday")
        return queryset

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super(MandayView, self).get_context_data(**kwargs)
        if 'month' in self.kwargs:
            context['month'] = self.kwargs['month']
            context['year'] = self.kwargs['year']
        else:
            context['month'] = datetime.date.today().month
            context['year'] = datetime.date.today().year
        return context


class ConsolidatedMandayView(LoginRequiredMixin, ListView):
    template_name = "projectdata/manday_list.html"
    context_object_name = "todolists"
    login_url = '/homelogin/'
    redirect_field_name = 'redirect_to'

    def get_queryset(self):
        if 'month' in self.kwargs:
            month = self.kwargs['month']
            year = self.kwargs['year']
        else:
            month = datetime.date.today().month
            year = datetime.date.today().year
        if self.request.user.hod:
            queryset = Man_Hours_model.objects.all()

        elif self.request.user.hos:
            queryset = Man_Hours_model.objects.filter(user__section=self.request.user.section)
        elif self.request.user.inCharge:
            queryset = Man_Hours_model.objects.filter(user__system=self.request.user.system)
        else:
            queryset = Man_Hours_model.objects.filter(user=self.request.user)

        if month == 0 and year == 0:
            queryset = queryset
        elif month == 0:
            queryset = queryset.filter(year=year)
        elif year == 0:
            queryset = queryset.filter(month=month)
        else:
            queryset = queryset.filter(month=month, year=year)
        queryset = queryset.order_by('user__username', 'date')
        print(self.request.user.username + " Viewed Consolidated Man-day sheet")
        return queryset

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super(ConsolidatedMandayView, self).get_context_data(**kwargs)
        if 'month' in self.kwargs:
            context['month'] = self.kwargs['month']
            context['year'] = self.kwargs['year']
        else:
            context['month'] = datetime.date.today().month
            context['year'] = datetime.date.today().year
        return context


@login_required(login_url="/homelogin/")
def Manday_create(request, *args, **kwargs):
    if 'month' in kwargs:
        if kwargs['month'] != 0:
            month = kwargs['month']

        else:
            pass
    else:
        month = datetime.date.today().month
    queryset = ListOfItems.objects.filter(Q(assigned_to=request.user,
                                            endDate__month=datetime.date.today().month,
                                            endDate__year=datetime.date.today().year,
                                            mandayCreated=False) | Q(assigned_to=request.user,
                                                                     startDate__month=datetime.date.today().month,
                                                                     startDate__year=datetime.date.today().year,
                                                                     mandayCreated=False)).exclude(
        status='Not required').order_by('assigned_to', 'startDate')
    timequery = Time_Management.objects.filter(user=request.user, date__date__month=datetime.date.today().month)
    print(request.user.username + " Creating Man-day")
    for item in queryset:
        item.mandayCreated = True
        item.save()
        datediff = item.endDate - item.startDate
        for i in range(0, datediff.days + 1, 1):
            if Dates.objects.get(date=item.startDate + timedelta(days=i)) not in item.excludeDates.all():
                if i == datediff.days:
                    if ListOfItems.objects.get(pk=item.pk).project.project_code == "Education":
                        new_manday = Man_Hours_model.objects.create(
                            date=(item.startDate + datetime.timedelta(days=i)).day,
                            month=(item.startDate + datetime.timedelta(days=i)).month,
                            year=(item.startDate + datetime.timedelta(days=i)).year,
                            project=ListOfItems.objects.get(pk=item.pk),
                            startDate=item.startDate + timedelta(days=i),
                            endDate=item.endDate,
                            user=request.user,
                        )
                    else:
                        new_manday = Man_Hours_model.objects.create(
                            date=(item.startDate + datetime.timedelta(days=i)).day,
                            month=(item.startDate + datetime.timedelta(days=i)).month,
                            year=(item.startDate + datetime.timedelta(days=i)).year,
                            project=ListOfItems.objects.get(pk=item.pk),
                            startDate=item.startDate,
                            endDate=item.endDate,
                            user=request.user)
                else:
                    if ListOfItems.objects.get(pk=item.pk).project.project_code == "Education":
                        new_manday = Man_Hours_model.objects.create(
                            date=(item.startDate + datetime.timedelta(days=i)).day,
                            month=(item.startDate + datetime.timedelta(days=i)).month,
                            year=(item.startDate + datetime.timedelta(days=i)).year,
                            project=ListOfItems.objects.get(pk=item.pk),
                            startDate=item.startDate + timedelta(days=i),
                            endDate=item.startDate + timedelta(days=i),
                            user=request.user)
                    else:
                        new_manday = Man_Hours_model.objects.create(
                            date=(item.startDate + datetime.timedelta(days=i)).day,
                            month=(item.startDate + datetime.timedelta(days=i)).month,
                            year=(item.startDate + datetime.timedelta(days=i)).year,
                            project=ListOfItems.objects.get(pk=item.pk),
                            startDate=item.startDate,
                            endDate=None,
                            user=request.user)
                if Dates.objects.get(date=datetime.date(day=new_manday.date, month=new_manday.month,
                                                        year=new_manday.startDate.year)).holiday:
                    new_manday.holiday_working = True
                    new_manday.save()
    queryset = Man_Hours_model.objects.filter(user=request.user, month=datetime.date.today().month)
    unique_dates = queryset.values('date').distinct()
    for day in unique_dates:
        numberOfItems = queryset.filter(date=day['date'])
        if len(timequery.filter(date__date__day=day['date'])) > 0:
            if timequery.filter(date__date__day=day['date'])[0].intime and \
                    timequery.filter(date__date__day=day['date'])[0].outtime:
                starttime = datetime.datetime.combine(datetime.date.today(),
                                                      timequery.filter(date__date__day=day['date'])[0].intime)
                endtime = datetime.datetime.combine(datetime.date.today(),
                                                    timequery.filter(date__date__day=day['date'])[0].outtime)
                reminder = endtime - starttime
                if timequery.filter(date__date__day=day['date'])[0].breakfast:
                    reminder = reminder - datetime.timedelta(minutes=30)
                if timequery.filter(date__date__day=day['date'])[0].lunch:
                    reminder = reminder - datetime.timedelta(minutes=30)
                x = reminder
                reminder = round(x.total_seconds() / 3600, 2)
                x = reminder
            else:
                x = 8.5
                reminder = 8.5

        else:
            reminder = 8.5
            x = 8.5
        for i in range(0, len(numberOfItems)):
            if i < len(numberOfItems) - 1:
                DefaultManhour = round(x / len(numberOfItems), 2)
                reminder = reminder - DefaultManhour
            if i == len(numberOfItems) - 1:
                DefaultManhour = round(reminder, 2)
            item = numberOfItems[i]
            if not item.hoursGenerated:
                item.workhours = DefaultManhour
                item.workhours_time = datetime.timedelta(hours=int(DefaultManhour),
                                                         minutes=60 * (DefaultManhour - int(DefaultManhour)))
                item.hoursGenerated = True
                item.save()
    return HttpResponseRedirect(reverse('manday-view'))


@login_required(login_url="/homelogin/")
def Manday_edit(request, pk):
    context = {}
    queryset = Man_Hours_model.objects.get(pk=pk)
    form = MandayEditForm(request.POST or None, instance=queryset)
    if form.is_valid():
        form.save()
        return HttpResponseRedirect(reverse("manday-detail", kwargs={'pk': pk}))
    context['form'] = form
    context['projects'] = queryset
    print(request.user.username + " Editing Man-day")
    return render(request, "partials/manhours_edit.html", context)


@login_required(login_url="/homelogin/")
def Manday_delete(request):
    queryset = Man_Hours_model.objects.filter(user=request.user, month=datetime.date.today().month,
                                              year=datetime.date.today().year)
    for item in queryset:
        item.project.mandayCreated = False
        item.project.save()
        item.delete()
    resetmanday = ListOfItems.objects.filter(assigned_to=request.user, endDate__month__gte=datetime.date.today().month,
                                             endDate__year__gte=datetime.date.today().year)
    for item in resetmanday:
        item.mandayCreated = False
        item.save()
    print(request.user.username + " Deleting Manday")
    return HttpResponseRedirect(reverse('manday-view'))


@login_required(login_url="/homelogin/")
def Manday_detail(request, pk):
    manday = Man_Hours_model.objects.get(pk=pk)
    context = {"todolist": manday}
    return render(request, "partials/manhours_detail_form.html", context)


@login_required(login_url="/homelogin/")
def download_manday(request, *args, **kwargs):
    if 'month' in kwargs:
        if kwargs['month'] != 0:
            month = kwargs['month']
        else:
            pass
    else:
        month = datetime.date.today().month
    if request.user.hos or request.user.hod:
        queryset = Man_Hours_model.objects.filter(month=month, year=datetime.date.today().year)
    elif request.user.inCharge:
        queryset = Man_Hours_model.objects.filter(month=month, user__system=request.user.system,
                                                  year=datetime.date.today().year)
    else:
        queryset = Man_Hours_model.objects.filter(month=month, user=request.user, year=datetime.date.today().year)

    queryset = queryset.order_by('user', 'date', )
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="Mandaysheet.csv"'
    writer = csv.writer(response)
    for query in queryset:
        writer.writerow([query.date,
                         query.project.subsystem,
                         query.project.project.project_code,
                         query.project.project.project_stage,
                         query.project.project.spec,
                         "worked on " + query.project.subsystem.name + " of " + query.project.project.project_code + "_" + query.project.project.project_stage,
                         query.project.remarks,
                         [user.first_name for user in query.project.assigned_to.all()],
                         query.startDate,
                         query.endDate,
                         query.workhours_time,
                         query.holiday_working])
    print(request.user.username + "Downloading Consolidated Manday")
    return response


@login_required(login_url="/homelogin/")
def download_manday_self(request, *args, **kwargs):
    if 'month' in kwargs:
        if kwargs['month'] != 0:
            month = kwargs['month']
            year = kwargs['year']

        else:
            pass
    else:
        month = datetime.date.today().month
        year = datetime.date.today().year
    queryset = Man_Hours_model.objects.filter(month=month, user=request.user, year=year)
    queryset = queryset.order_by('date')
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="file.csv"'
    writer = csv.writer(response)
    for query in queryset:
        writer.writerow([query.date,
                         query.project.subsystem,
                         query.project.project.project_code,
                         query.project.project.project_stage,
                         query.project.project.spec,
                         "worked on " + query.project.subsystem.name + " of " + query.project.project.project_code + "_" + query.project.project.project_stage,
                         query.project.remarks,
                         [user.first_name for user in query.project.assigned_to.all()],
                         query.startDate,
                         query.endDate,
                         query.workhours_time,
                         query.holiday_working])
    print(request.user.username + "Downloading Self Manday")
    return response


class WeeklyReportView(LoginRequiredMixin, ListView):
    template_name = "projectdata/weekly_report.html"
    context_object_name = "todolists"
    login_url = '/homelogin/'
    redirect_field_name = 'redirect_to'
    form_class = DateForm
    startDate = datetime.date.today()
    endDate = datetime.date.today()

    def get_queryset(self):
        if 'startDate' in self.kwargs:
            startDate = self.kwargs['startDate']
        else:
            startDate = datetime.date.today()
        if 'endDate' in self.kwargs:
            endDate = self.kwargs['endDate']
        else:
            endDate = datetime.date.today()

        if self.request.user.hos or self.request.user.hod or self.request.user.inCharge:
            queryset = ListOfItems.objects.filter(endDate__gte=startDate, endDate__lte=endDate).order_by(
                'project').exclude(status='Not required')
            queryset = queryset.values('project', 'project__project_code', 'project__project_stage', 'project__spec',
                                       'system__name', 'typeOfWork').annotate(total_issues=Sum('issueCount'),
                                                                              end_date=Max('endDate'))
            queryset = queryset.order_by('project__project_code', 'project__project_stage', 'typeOfWork',
                                         'system__name')
        print(self.request.user.username + " Viewed Weekly Report")
        return queryset

    def post(self, request, *args, **kwargs):
        self.startDate = self.request.POST.get('startDate')
        self.endDate = self.request.POST.get('endDate')
        return HttpResponseRedirect(self.get_success_url())

    def get_success_url(self):
        return reverse("weeklyreport-view-weekly", kwargs={'startDate': self.startDate,
                                                           'endDate': self.endDate})

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super(WeeklyReportView, self).get_context_data(**kwargs)
        context['form'] = self.form_class
        clash_check = {}
        cv = {}
        for q in self.object_list:
            if q['system__name'] != 'CV':
                if (q['project__project_code'], q['project__project_stage']) in clash_check.keys():
                    if q['typeOfWork'] in clash_check[(q['project__project_code'], q['project__project_stage'])]:
                        clash_check[(q['project__project_code'], q['project__project_stage'])][q['typeOfWork']][
                            q['system__name']] = q['total_issues']
                        clash_check[(q['project__project_code'], q['project__project_stage'])][q['typeOfWork']][
                            'end_date'] = q['end_date']
                        clash_check[(q['project__project_code'], q['project__project_stage'])][q['typeOfWork']][
                            'Total'] += q['total_issues']
                    else:
                        clash_check[(q['project__project_code'], q['project__project_stage'])][q['typeOfWork']] = {}
                        clash_check[(q['project__project_code'], q['project__project_stage'])][q['typeOfWork']][q[
                            'system__name']] = q['total_issues']
                        clash_check[(q['project__project_code'], q['project__project_stage'])][q['typeOfWork']][
                            'end_date'] = q['end_date']
                        clash_check[(q['project__project_code'], q['project__project_stage'])][q['typeOfWork']][
                            'Total'] = q['total_issues']
                else:
                    clash_check[(q['project__project_code'], q['project__project_stage'])] = {}
                    clash_check[(q['project__project_code'], q['project__project_stage'])][q['typeOfWork']] = {}
                    clash_check[(q['project__project_code'], q['project__project_stage'])][q['typeOfWork']][
                        q['system__name']] = q['total_issues']
                    clash_check[(q['project__project_code'], q['project__project_stage'])][q['typeOfWork']][
                        'end_date'] = q[
                        'end_date']
                    clash_check[(q['project__project_code'], q['project__project_stage'])][q['typeOfWork']][
                        'Total'] = q['total_issues']
            else:
                if (q['project__project_code'], q['project__project_stage'], q['project__spec']) in cv.keys():
                    if q['typeOfWork'] in cv[
                        (q['project__project_code'], q['project__project_stage'], q['project__spec'])]:
                        cv[(q['project__project_code'], q['project__project_stage'], q['project__spec'])][
                            q['typeOfWork']][q['system__name']] = q['total_issues']
                        cv[(q['project__project_code'], q['project__project_stage'], q['project__spec'])][
                            q['typeOfWork']]['end_date'] = q['end_date']
                    else:
                        cv[(q['project__project_code'], q['project__project_stage'], q['project__spec'])][
                            q['typeOfWork']] = {}
                        cv[(q['project__project_code'], q['project__project_stage'], q['project__spec'])][
                            q['typeOfWork']][q['system__name']] = q['total_issues']
                        cv[(q['project__project_code'], q['project__project_stage'], q['project__spec'])][
                            q['typeOfWork']]['end_date'] = q['end_date']
                else:
                    cv[(q['project__project_code'], q['project__project_stage'], q['project__spec'])] = {}
                    cv[(q['project__project_code'], q['project__project_stage'], q['project__spec'])][
                        q['typeOfWork']] = {}
                    cv[(q['project__project_code'], q['project__project_stage'], q['project__spec'])][q['typeOfWork']][
                        q['system__name']] = q['total_issues']
                    cv[(q['project__project_code'], q['project__project_stage'], q['project__spec'])][q['typeOfWork']][
                        'end_date'] = q['end_date']
        context['clash_check'] = clash_check
        context['cv'] = cv
        return context


class WorkDetailsView(LoginRequiredMixin, ListView):
    template_name = "projectdata/workdetails.html"
    context_object_name = "todolists"
    login_url = '/homelogin/'
    redirect_field_name = 'redirect_to'

    def get_queryset(self):
        if 'month' in self.kwargs:
            month = self.kwargs['month']
            year = self.kwargs['year']

        else:
            month = datetime.date.today().month
            year = datetime.date.today().year

        if self.request.user.hod:
            queryset = ListOfItems.objects.all().order_by(
                'project__receivedDate').exclude(status="Not required").exclude(project__project_code="Education") \
                .exclude(subsystem__typeOfWork="VR Kinematics")
        elif self.request.user.hos:
            queryset = ListOfItems.objects.filter(created_by__section=self.request.user.section).order_by(
                'project__receivedDate').exclude(status="Not required").exclude(project__project_code="Education") \
                .exclude(subsystem__typeOfWork="VR Kinematics")

        else:
            queryset = ListOfItems.objects.filter(system=self.request.user.system).order_by('project__receivedDate'). \
                exclude(status="Not required").exclude(project__project_code="Education"). \
                exclude(subsystem__typeOfWork="VR Kinematics")
        if year == 0 and month == 0:
            queryset = queryset
        elif month == 0:
            queryset = queryset.filter(project__receivedDate__year=year)
        elif year == 0:
            queryset = queryset.filter(project__receivedDate__month=month)
        else:
            queryset = queryset.filter(project__receivedDate__month=month, project__receivedDate__year=year)
        queryset = queryset.values('project__project_code',
                                   'project__project_stage',
                                   'project__spec',
                                   'project__company',
                                   'project__center',
                                   'system__name',
                                   'subsystem__name',
                                   'subsystem__typeOfWork',
                                   'project__receivedDate',
                                   'project__dueDate').distinct()
        queryset = queryset.order_by('project__receivedDate', 'project__project_code', 'project__project_stage',
                                     'subsystem__name')
        print(self.request.user.username + " Viewed Work Details")
        return queryset

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super(WorkDetailsView, self).get_context_data(**kwargs)
        if 'month' in self.kwargs:
            context['month'] = self.kwargs['month']
            context['year'] = self.kwargs['year']
        else:
            context['month'] = datetime.date.today().month
            context['year'] = datetime.date.today().year
        return context


class ProjectDetailsView(LoginRequiredMixin, ListView):
    template_name = "projectdata/project_completion.html"
    context_object_name = "todolists"
    login_url = '/homelogin/'
    redirect_field_name = 'redirect_to'

    def get_queryset(self):
        if 'month' in self.kwargs:
            if self.kwargs['month'] != 0:
                queryset = ListOfItems.objects.filter(project__receivedDate__month=self.kwargs['month'],
                                                      project__receivedDate__year=self.kwargs['year']).exclude(
                    status="Not required").exclude(typeOfWork="Education")

            elif self.kwargs['month'] == 0:
                queryset = ListOfItems.objects.filter(project__receivedDate__year=self.kwargs['year']).exclude(
                    status="Not required").exclude(typeOfWork="Education")
        else:
            queryset = ListOfItems.objects.filter(project__receivedDate__month=datetime.date.today().month,
                                                  project__receivedDate__year=datetime.date.today().year).exclude(
                status="Not required").exclude(typeOfWork="Education")
        return queryset

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super(ProjectDetailsView, self).get_context_data(**kwargs)

        if 'month' in self.kwargs:
            context['month'] = self.kwargs['month']
            context['year'] = self.kwargs['year']
        else:
            context['month'] = datetime.date.today().month
            context['year'] = datetime.date.today().year
        context['Allworks'] = self.object_list.filter(project__spec=None).exclude(system__name='CV').values(
            'project__project_code',
            'project__project_stage', 'typeOfWork').distinct(
        ).order_by('typeOfWork', 'project__project_code')
        context['clash_check'] = self.object_list.filter(typeOfWork="Clash Check").exclude(system__name='CV').values(
            'project__project_code', 'project__project_stage').distinct().order_by('project__project_code')
        context['kinematics'] = self.object_list.filter(typeOfWork="kinematics").exclude(system__name='CV').values(
            'project__project_code', 'project__project_stage').distinct().order_by('project__project_code')
        context['IPS'] = self.object_list.filter(typeOfWork="IPS").exclude(system__name='CV').values(
            'project__project_code', 'project__project_stage').distinct().order_by('project__project_code')
        context['VR_Modeling'] = self.object_list.filter(typeOfWork="VR Modeling").exclude(system__name='CV').values(
            'project__project_code', 'project__project_stage').distinct().order_by('project__project_code')
        context['CFD'] = self.object_list.filter(typeOfWork="CFD").exclude(system__name='CV').values(
            'project__project_code', 'project__project_stage').distinct().order_by('project__project_code')
        context['Over_Travel'] = self.object_list.filter(typeOfWork="Over Travel").exclude(system__name='CV').values(
            'project__project_code', 'project__project_stage').distinct().order_by('project__project_code')
        context['SubMaterial'] = self.object_list.filter(typeOfWork="SubMaterial").exclude(system__name='CV').values(
            'project__project_code', 'project__project_stage').distinct().order_by('project__project_code')
        context['WeldCheck'] = self.object_list.filter(typeOfWork="WeldCheck").exclude(system__name='CV').values(
            'project__project_code', 'project__project_stage').distinct().order_by('project__project_code')
        context['Hole_HW_Check'] = self.object_list.filter(typeOfWork="Hole HW Check").exclude(
            system__name='CV').values('project__project_code', 'project__project_stage').distinct().order_by(
            'project__project_code')
        context['BSR_Check'] = self.object_list.filter(typeOfWork="BSR Check").exclude(system__name='CV').values(
            'project__project_code', 'project__project_stage').distinct().order_by('project__project_code')
        context['Install_drawing_Analysis'] = self.object_list.filter(typeOfWork="Install drawing Analysis").exclude(
            system__name='CV').values('project__project_code', 'project__project_stage').distinct().order_by(
            'project__project_code')
        context['Assembly_simulation'] = self.object_list.filter(typeOfWork="Assembly simulation").exclude(
            system__name='CV').values('project__project_code', 'project__project_stage').distinct().order_by(
            'project__project_code')
        context['CV'] = self.object_list.filter(system__name='CV').values('project__project_code',
                                                                          'project__project_stage', 'project__spec',
                                                                          'typeOfWork').distinct().order_by(
            'typeOfWork', 'project__project_code')

        print(self.request.user.username + " Viewed Project Details")
        return context


class FinanceSheetView(LoginRequiredMixin, ListView):
    template_name = "projectdata/finance_sheet.html"
    context_object_name = "manday"
    login_url = '/homelogin/'
    redirect_field_name = 'redirect_to'

    def get_queryset(self):
        if 'month' in self.kwargs:
            if self.kwargs['month'] != 0:
                month = self.kwargs['month']
                year = self.kwargs['year']
            else:
                pass
        else:
            month = datetime.date.today().month
            year = datetime.date.today().year
        queryset = Man_Hours_model.objects.filter(month=month, year=year).order_by('user__first_name')
        return queryset

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super(FinanceSheetView, self).get_context_data(**kwargs)
        if 'month' in self.kwargs:
            context['month'] = self.kwargs['month']
            context['year'] = self.kwargs['year']
        else:
            context['month'] = datetime.date.today().month
            context['year'] = datetime.date.today().year
        print(self.request.user.username + " Viewed Finance Sheet")
        context['HMC'] = self.object_list.filter(project__project__company='HMC')
        context['KMC'] = self.object_list.filter(project__project__company='KMC')
        context['HMIE'] = self.object_list.filter(project__project__company='HMIE')
        HMC_manhours = datetime.timedelta(hours=0, minutes=0)
        KMC_manhours = datetime.timedelta(hours=0, minutes=0)
        HMIE_manhours = datetime.timedelta(hours=0, minutes=0)
        for item in context['HMC']:
            HMC_manhours += item.workhours_time
        for item in context['KMC']:
            KMC_manhours += item.workhours_time
        for item in context['HMIE']:
            HMIE_manhours += item.workhours_time

        context['HMC'] = HMC_manhours
        context['KMC'] = KMC_manhours
        context['HMIE'] = HMIE_manhours
        return context


def hrdesk_request(request):
    ip = request.META.get("REMOTE_ADDR")
    sess = requests.Session()
    sess.auth = ('4014065', 'Aatrox@1234')
    auth = sess.post("https://hmieesprd.hmie.co.in/ess/", auth=HTTPBasicAuth('4014065', 'Aatrox@1234'))
    resp = sess.get("https://hmieesprd.hmie.co.in/ess/app.html#/tm/flexihrsweeklyreport?startDate=2023-01-12")
    print(resp.text)
    print(resp.json())
    return HttpResponseRedirect(reverse('Home'))


@login_required(login_url="/homelogin/")
def createClashCheck(request, pk):
    tasks = SubSystem.objects.filter(system=request.user.system).exclude(typeOfWork="Education")
    queryset = ListOfItems.objects.filter(project=pk, system=request.user.system)
    for query in queryset:
        tasks = tasks.exclude(name=query.subsystem)
    project = Project.objects.get(pk=pk)
    for task in tasks:
        defaulttask = ListOfItems.objects.create(project=project,
                                                 system=request.user.system,
                                                 status="On going",
                                                 subsystem=task,
                                                 typeOfWork=task.typeOfWork,
                                                 startDate=project.receivedDate,
                                                 endDate=project.dueDate,
                                                 created_by=request.user, )
        defaulttask.save()
    return HttpResponseRedirect(reverse('view-project', kwargs={'pk': pk}))


@login_required(login_url="/homelogin/")
def setNotRequired(request, pk):
    queryset = ListOfItems.objects.filter(project=pk, system=request.user.system, assigned_to=None).exclude(
        status="Finished")
    for query in queryset:
        query.status = "Not required"
        query.save()
    return HttpResponseRedirect(reverse('view-project', kwargs={'pk': pk}))


@login_required(login_url="/homelogin/")
def deleteUnset(request, pk):
    print(request.user.username + " Deleting Unset tasks")
    if request.user.inCharge:
        queryset = ListOfItems.objects.filter(project=pk, system=request.user.system, assigned_to=None).exclude(
            status="Finished"). \
            exclude(status='Not required')
        for query in queryset:
            query.delete()
        return HttpResponseRedirect(reverse('view-project', kwargs={'pk': pk}))
    else:
        return HttpResponse('Unauthorized', status=401)


@login_required(redirect_field_name="/homecalender/", login_url="/homelogin/")
def renderCalander(request):
    print(request.user.username + " Viewed work calender")
    queryset = ListOfItems.objects.filter(Q(assigned_to=request.user,
                                            endDate__month__gte=datetime.date.today().month,
                                            endDate__year=datetime.date.today().year) | Q(assigned_to=request.user,
                                                                                          startDate__month__gte=datetime.date.today().month,
                                                                                          startDate__year=datetime.date.today().year)).exclude(
        status="Not required").prefetch_related('excludeDates')
    dates_query = Dates.objects.filter(
        date__lte=datetime.date(day=calendar.monthrange(datetime.date.today().year, datetime.date.today().month)[1],
                                month=datetime.date.today().month,
                                year=datetime.date.today().year),
        date__gte=datetime.date(day=1,
                                month=datetime.date.today().month,
                                year=datetime.date.today().year))
    time_query = Time_Management.objects.filter(Q(user=request.user,
                                                  date__date__month=datetime.date.today().month,
                                                  date__date__year=datetime.date.today().year) | Q(user=request.user,
                                                                                                   date__date__month=datetime.date.today().month - 1,
                                                                                                   date__date__year=datetime.date.today().year))

    aggregate_query = time_query.annotate(week=ExtractWeek('date__date')).values('week').annotate(
        week_hours=Sum('worktime')).values('week', 'week_hours')
    for item in aggregate_query:
        time1 = item['week_hours'].total_seconds()
        item['leave'] = 0
        item['hours'], item['minutes'], item['seconds'] = time1 // 3600, (time1 % 3600) // 60, time1 % 60
        x = datetime.timedelta(hours=0, minutes=0)
        for query in time_query.filter(date__date__week=item['week']).exclude(date__holiday=True):
            if query.intime:
                if query.intime > datetime.time(hour=9, minute=30):
                    x = x - datetime.timedelta(hours=4, minutes=30)
                    item['leave'] += 0.5
            if query.outtime:
                if query.outtime < datetime.time(hour=16, minute=00):
                    x = x - datetime.timedelta(hours=4, minutes=00)
                    item['leave'] += 0.5
            x = x + datetime.timedelta(hours=8, minutes=30)
        if x > datetime.timedelta(hours=42, minutes=30):
            x = datetime.timedelta(hours=42, minutes=30)
        item['eligible_hours'], item['eligible_minutes'] = x.total_seconds() // 3600, (x.total_seconds() % 3600) // 60
        if x.total_seconds() - time1 < 0:
            item['pending_hours'], item['pending_minutes'] = (time1 - x.total_seconds()) // 3600, ((
                                                                                                           time1 - x.total_seconds()) % 3600) // 60
            item['value'] = -1
        else:
            item['pending_hours'], item['pending_minutes'] = (x.total_seconds() - time1) // 3600, ((
                                                                                                           x.total_seconds() - time1) % 3600) // 60
            item['value'] = 1

    context = {'data': queryset,
               'times': time_query,
               'stddates': dates_query.values('date', 'holiday', 'date__iso_week_day', 'date__week'),
               'dates': dates_query,
               'weekhours': aggregate_query,
               'today':datetime.date.today()}
    return render(request, "partials/calender_1.html", context)



@login_required(redirect_field_name="/homecalender/", login_url="/homelogin/")
def calculateTimes(request,weeknumber):
    try:
        weeknumber = datetime.strptime(weeknumber,"%m/%d/%y")
        weeknumber = weeknumber.isocalendar()[1]
    except:
        weeknumber=weeknumber
    time_query = Time_Management.objects.filter(Q(user=request.user,date__date__month=datetime.date.today().month,
                                                  date__date__year=datetime.date.today().year) | Q(user=request.user,
                                                                                                   date__date__month=datetime.date.today().month - 1,
                                                                                                   date__date__year=datetime.date.today().year))
    aggregate_query = time_query.annotate(week=ExtractWeek('date__date')).values('week').annotate(
        week_hours=Sum('worktime')).values('week', 'week_hours').filter(week=weeknumber)
    if len(aggregate_query)>0:
        item = aggregate_query[0]
        time1 = item['week_hours'].total_seconds()
        item['leave'] = 0
        item['hours'], item['minutes'], item['seconds'] = time1 // 3600, (time1 % 3600) // 60, time1 % 60
        x = datetime.timedelta(hours=0, minutes=0)
        for query in time_query.filter(date__date__week=weeknumber).exclude(date__holiday=True):
            if query.intime:
                if query.intime > datetime.time(hour=9, minute=30):
                    x = x - datetime.timedelta(hours=4, minutes=30)
                    item['leave'] += 0.5
            if query.outtime:
                if query.outtime < datetime.time(hour=16, minute=00):
                    x = x - datetime.timedelta(hours=4, minutes=00)
                    item['leave'] += 0.5
            x = x + datetime.timedelta(hours=8, minutes=30)
        if x > datetime.timedelta(hours=42, minutes=30):
            x = datetime.timedelta(hours=42, minutes=30)
        item['eligible_hours'], item['eligible_minutes'] = x.total_seconds() // 3600, (x.total_seconds() % 3600) // 60
        if x.total_seconds() - time1 < 0:
            item['pending_hours'], item['pending_minutes'] = (time1 - x.total_seconds()) // 3600, ((time1 - x.total_seconds()) % 3600) // 60
            item['value'] = -1
        else:
            item['pending_hours'], item['pending_minutes'] = (x.total_seconds() - time1) // 3600, ((x.total_seconds() - time1) % 3600) // 60
            item['value'] = 1
    else:
        item={'week': weeknumber,
              'week_hours': datetime.timedelta(days=0, seconds=0),
              'leave': 0,
              'hours': 0,
              'minutes': 0,
              'seconds': 0.0,
              'eligible_hours': 0,
              'eligible_minutes': 0,
              'pending_hours': 0,
              'pending_minutes': 0,
              'value': -1}
    context = {'weekhours': item}
    return render(request, "partials/total_work_time_form.html", context)


@login_required(redirect_field_name="/homecalender/", login_url="/homelogin/")
def intimecreate(request, date):
    initial = {'intime': datetime.datetime.now()}
    form = TimeManagementInTimeForm(request.POST or None)
    context = {"form": form, 'date': date}
    date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
    print(request.user.username + " Created Intime")
    if form.is_valid():
        intime = form.save(commit=False)
        intime.user = request.user
        intime.date = Dates.objects.get(date=date)
        query = Time_Management.objects.filter(user=request.user, date=intime.date)
        if len(query) == 0:
            intime.save()
            return HttpResponseRedirect(reverse("detail-time", kwargs={'pk': intime.id}))
        else:
            query[0].intime = intime.intime
            query[0].save()
            return HttpResponseRedirect(reverse("detail-time", kwargs={'pk': query[0].pk}))
    return render(request, "partials/time_increate.html", context)


@login_required(redirect_field_name="/homecalender/", login_url="/homelogin/")
def outtimecreate(request, date):
    initial = {'outtime': datetime.datetime.now()}
    form = TimeManagementOutTimeForm(request.POST or None)
    context = {"form": form, 'date': date}
    date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
    print(request.user.username + " Created Outtime")
    if request.method == "POST":
        if form.is_valid():
            outtime = form.save(commit=False)
            outtime.user = request.user
            outtime.date = Dates.objects.get(date=date)
            query = Time_Management.objects.filter(user=request.user, date=outtime.date)
            if len(query) == 0:
                outtime.save()
                return HttpResponseRedirect(reverse("detail-outtime", kwargs={'pk': outtime.id}))
            else:
                query[0].outtime = outtime.outtime
                query[0].save()
                return HttpResponseRedirect(reverse("detail-outtime", kwargs={'pk': query[0].pk}))
    return render(request, "partials/time_outcreate.html", context)


@login_required(redirect_field_name="/homecalender/", login_url="/homelogin/")
def datetimeview(request, pk):
    work = Time_Management.objects.get(pk=pk)
    context = {"time": work}
    response = render(request, "partials/time_detail_form.html", context)
    return trigger_client_event(response,"time_update")

@login_required(redirect_field_name="/homecalender/", login_url="/homelogin/")
def datetimeoutview(request, pk):
    work = Time_Management.objects.get(pk=pk)
    context = {"time": work}
    response = render(request, "partials/time_outdetail_form.html", context)
    return trigger_client_event(response,"time_update")

@login_required(redirect_field_name="/homecalender/", login_url="/homelogin/")
def update_intime(request, pk):
    context = {}
    queryset = Time_Management.objects.get(pk=pk)
    form = TimeManagementInTimeForm(request.POST or None, instance=queryset)
    if form.is_valid():
        form1 = form.save(commit=False)
        form1.outtime = queryset.outtime
        form1.user = request.user
        form1.date = queryset.date
        form1.save()
        response = HttpResponseRedirect(reverse("detail-time", kwargs={'pk': pk}))
        return response
    context['form'] = form
    context['work'] = queryset
    context['date'] = queryset.date.date
    print(request.user.username + " Updated in-time")
    return render(request, "partials/time_inedit.html", context)


@login_required(redirect_field_name="/homecalender/", login_url="/homelogin/")
def update_outtime(request, pk):
    context = {}
    queryset = Time_Management.objects.get(pk=pk)
    form = TimeManagementOutTimeForm(request.POST or None, instance=queryset)
    if form.is_valid():
        form1 = form.save(commit=False)
        form1.intime = queryset.intime
        form1.user = request.user
        form1.date = queryset.date
        form1.save()
        response = HttpResponseRedirect(reverse("detail-outtime", kwargs={'pk': pk}))
        return response
    context['form'] = form
    context['work'] = queryset
    context['date'] = queryset.date.date
    print(request.user.username + " Updated out-time")
    response = render(request, "partials/time_outedit.html", context)
    return response
