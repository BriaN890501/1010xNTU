# Generated by Django 3.1.5 on 2021-02-01 11:14

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('input_output_preview', '0002_auto_20210201_1113'),
    ]

    operations = [
        migrations.RenameField(
            model_name='method',
            old_name='NameError',
            new_name='Name',
        ),
    ]
