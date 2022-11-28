# Generated by Django 4.0.4 on 2022-09-11 17:04

import django.contrib.postgres.fields
import django.core.validators
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ("users", "0002_user_allow_logging_user_preferred_taxonomies"),
        ("citation_screening", "0006_remove_literaturereviewmember_user_and_more"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="literaturereview",
            name="first_screening",
        ),
        migrations.AddField(
            model_name="literaturereview",
            name="additional_description",
            field=models.TextField(
                blank=True,
                default="",
                help_text="Additional description field, for instance for topic narrative.",
                null=True,
            ),
        ),
        migrations.AddField(
            model_name="literaturereview",
            name="annotations_per_paper",
            field=models.IntegerField(
                default=1,
                help_text="How many reviewers need to screen every paper. Default is 1.",
                validators=[
                    django.core.validators.MinValueValidator(1),
                    django.core.validators.MaxValueValidator(3),
                ],
            ),
        ),
        migrations.AddField(
            model_name="literaturereview",
            name="creation_date",
            field=models.DateField(
                auto_now_add=True, default=django.utils.timezone.now
            ),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="literaturereview",
            name="discipline",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                to="users.knowledgearea",
            ),
        ),
        migrations.AddField(
            model_name="literaturereview",
            name="last_edit_date",
            field=models.DateField(auto_now=True),
        ),
        migrations.AddField(
            model_name="literaturereview",
            name="project_deadline",
            field=models.DateField(default=django.utils.timezone.now),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="literaturereview",
            name="search_databases",
            field=models.CharField(blank=True, max_length=250, null=True),
        ),
        migrations.AddField(
            model_name="literaturereview",
            name="tags",
            field=django.contrib.postgres.fields.ArrayField(
                base_field=models.CharField(blank=True, max_length=250),
                null=True,
                size=None,
            ),
        ),
        migrations.DeleteModel(
            name="CitationScreening",
        ),
    ]
