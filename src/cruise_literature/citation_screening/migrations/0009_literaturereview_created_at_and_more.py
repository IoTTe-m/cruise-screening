# Generated by Django 4.0.4 on 2022-10-24 15:04

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("citation_screening", "0008_alter_literaturereview_tags"),
    ]

    operations = [
        migrations.AddField(
            model_name="literaturereview",
            name="created_at",
            field=models.DateTimeField(
                auto_now_add=True,
                default=django.utils.timezone.now,
                verbose_name="created at",
            ),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="literaturereview",
            name="updated_at",
            field=models.DateTimeField(auto_now=True, verbose_name="updated at"),
        ),
        migrations.AddField(
            model_name="literaturereviewmember",
            name="created_at",
            field=models.DateTimeField(
                auto_now_add=True,
                default=django.utils.timezone.now,
                verbose_name="created at",
            ),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="literaturereviewmember",
            name="updated_at",
            field=models.DateTimeField(auto_now=True, verbose_name="updated at"),
        ),
        migrations.AlterField(
            model_name="literaturereviewmember",
            name="member",
            field=models.ForeignKey(
                help_text="User ID",
                on_delete=django.db.models.deletion.CASCADE,
                related_name="lrm_through",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
    ]
