#!/usr/bin/env python3
"""
Script to automatically add noise experiment plots to Google Slides presentation.
Uploads aggregated boxplots and trajectory plots for all models and conditions.
"""

import os
import glob
import time
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import pickle

# Scopes required for Google Slides and Drive APIs
SCOPES = [
    'https://www.googleapis.com/auth/presentations',
    'https://www.googleapis.com/auth/drive.file'
]

# Presentation ID extracted from the URL
PRESENTATION_ID = '1W0jz81TXkTHMixFNV89lsCsF_2nd9Kjxrue_bsAxEW4'

# Starting slide number (1-indexed)
START_SLIDE = 406

# Boxplot filenames (in display order)
BOXPLOT_FILES = [
    'condition_comparison.png',
    'cumulative_balance_boxplot_by_condition.png',
    'cumulative_balances_by_condition.png',
    'topic_comparison.png',
    'cumulative_balance_boxplot_by_topic.png',
    'cumulative_balances_by_topic.png',
]

# Balance comparison plot filenames
BALANCE_COMPARISON_PLOTS = [
    'balance_comparison.png',
    'delta_comparison.png',
]

# Summary plot filenames per task_order
SUMMARY_PLOTS = [
    'conditions_summary.png',
    'conditions_trajectories.png',
]

# Experiment definitions (v2 structure)
EXPERIMENTS = [
    {
        'label': 'GPT-5-Nano (Bootstrap)',
        'experiment': 'noise_bootstrap_mem3',
        'model': 'gpt-5-nano',
        'conditions': ['noisy_bootstrap_cooperation', 'noisy_bootstrap_cooperation_informed'],
        'task_orders': ['game', 'game_myth', 'myth_game'],
    },
    {
        'label': 'Gemini 3.1 Pro (Negative)',
        'experiment': 'noise_negative_mem3',
        'model': 'gemini-3.1-pro-preview',
        'conditions': ['noisy_negative_5', 'noisy_negative_5_informed'],
        'task_orders': ['game', 'game_myth', 'myth_game'],
    },
    {
        'label': 'Claude Sonnet 4.5 (Negative)',
        'experiment': 'noise_negative_mem3',
        'model': 'claude-sonnet-4.5',
        'conditions': ['noisy_negative_5', 'noisy_negative_5_informed'],
        'task_orders': ['game', 'game_myth', 'myth_game'],
    },
]

V2_BASE = 'data/plots/noise_experiments/v2_uniform_distribution_noise'


def authenticate():
    """Authenticate with Google API."""
    creds = None
    token_path = 'token.pickle'

    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)

    return creds


def get_noise_experiment_plots():
    """Collect all noise experiment plots in presentation order (v2 structure).

    Layout assumptions:
      {V2_BASE}/{experiment}/_balance_comparison/{model}/{balance_comparison,delta_comparison}.png
      {V2_BASE}/{experiment}/{model}/boxplot/*.png
      {V2_BASE}/{experiment}/{model}/{task_order}/{conditions_summary,conditions_trajectories}.png
      {V2_BASE}/{experiment}/{model}/{task_order}/{condition}/<run_name>.png  (per-run trajectory)
    """
    plots = []

    for exp in EXPERIMENTS:
        label = exp['label']
        experiment = exp['experiment']
        model = exp['model']
        exp_base = os.path.join(V2_BASE, experiment)
        model_base = os.path.join(exp_base, model)

        # 1. Balance comparison plots (per-experiment, per-model)
        balance_model_dir = os.path.join(exp_base, '_balance_comparison', model)
        if os.path.isdir(balance_model_dir):
            for plot_name in BALANCE_COMPARISON_PLOTS:
                path = os.path.join(balance_model_dir, plot_name)
                if os.path.exists(path):
                    display_name = plot_name.replace('.png', '').replace('_', ' ').title()
                    plots.append({
                        'path': path,
                        'title': f"{label} | Balance Comparison | {display_name}",
                    })
                    print(f"  Found balance comparison: {experiment} / {model} / {plot_name}")

        # 2. Per-model boxplots
        model_boxplots_dir = os.path.join(model_base, 'boxplot')
        if os.path.isdir(model_boxplots_dir):
            for plot_name in BOXPLOT_FILES:
                path = os.path.join(model_boxplots_dir, plot_name)
                if os.path.exists(path):
                    display_name = plot_name.replace('.png', '').replace('_', ' ').title()
                    plots.append({
                        'path': path,
                        'title': f"{label} | {display_name}",
                    })
                    print(f"  Found model boxplot: {model} / {plot_name}")

        # 3. Per task_order: summary plots + flattened per-run trajectory PNGs
        for task_order in exp['task_orders']:
            task_dir = os.path.join(model_base, task_order)
            if not os.path.isdir(task_dir):
                continue

            # Summary plots
            for plot_name in SUMMARY_PLOTS:
                path = os.path.join(task_dir, plot_name)
                if os.path.exists(path):
                    display_name = plot_name.replace('.png', '').replace('_', ' ').title()
                    plots.append({
                        'path': path,
                        'title': f"{label} | {task_order} | {display_name}",
                    })
                    print(f"  Found summary: {model} / {task_order} / {plot_name}")

            # Per-run trajectory plots (now flat PNGs directly under each condition dir)
            for condition in exp['conditions']:
                cond_dir = os.path.join(task_dir, condition)
                if not os.path.isdir(cond_dir):
                    continue

                for run_png in sorted(glob.glob(os.path.join(cond_dir, '*.png'))):
                    fname = os.path.basename(run_png)
                    # Skip the aggregate file that shares a generic name
                    if fname == 'trajectory_1_numerical.png':
                        continue
                    run_name = fname[:-len('.png')]
                    plots.append({
                        'path': run_png,
                        'title': f"{label} | {condition} | {task_order} | {run_name}",
                    })

        print(f"  {label}: collected so far = {len(plots)} total")

    return plots


def create_blank_slides(slides_service, presentation_id, count, insert_index):
    """Create N blank slides starting at insert_index."""
    print(f"Creating {count} blank slides at position {insert_index + 1}...")
    requests = []
    for i in range(count):
        requests.append({
            'createSlide': {
                'insertionIndex': insert_index + i,
                'slideLayoutReference': {'predefinedLayout': 'BLANK'}
            }
        })

    # Batch in groups of 50 to stay within API limits
    for batch_start in range(0, len(requests), 50):
        batch = requests[batch_start:batch_start + 50]
        slides_service.presentations().batchUpdate(
            presentationId=presentation_id,
            body={'requests': batch}
        ).execute()
        print(f"  Created slides {batch_start + 1}-{min(batch_start + 50, len(requests))}")


def upload_image_to_drive(service, image_path):
    """Upload image to Google Drive and return the file ID."""
    file_metadata = {
        'name': os.path.basename(image_path),
        'mimeType': 'image/png'
    }

    media = MediaFileUpload(image_path, mimetype='image/png', resumable=True)

    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id, webContentLink'
    ).execute()

    service.permissions().create(
        fileId=file['id'],
        body={'type': 'anyone', 'role': 'reader'}
    ).execute()

    return file['id']


def add_image_to_slide(slides_service, drive_service, presentation_id, slide_index, image_path, title_text):
    """Add an image and title to a specific slide."""
    presentation = slides_service.presentations().get(
        presentationId=presentation_id
    ).execute()

    slides = presentation.get('slides', [])

    if slide_index >= len(slides):
        print(f"Warning: Slide {slide_index + 1} doesn't exist. Presentation has {len(slides)} slides.")
        return False

    page_id = slides[slide_index]['objectId']

    print(f"  Uploading image: {os.path.basename(image_path)}")
    image_id = upload_image_to_drive(drive_service, image_path)
    image_url = f'https://drive.google.com/uc?id={image_id}'

    ts = int(time.time() * 1000)
    image_object_id = f'image_{ts}'
    title_object_id = f'title_{ts + 1}'

    requests = [
        {
            'createShape': {
                'objectId': title_object_id,
                'shapeType': 'TEXT_BOX',
                'elementProperties': {
                    'pageObjectId': page_id,
                    'size': {
                        'height': {'magnitude': 50, 'unit': 'PT'},
                        'width': {'magnitude': 600, 'unit': 'PT'}
                    },
                    'transform': {
                        'scaleX': 1,
                        'scaleY': 1,
                        'translateX': 50,
                        'translateY': 30,
                        'unit': 'PT'
                    }
                }
            }
        },
        {
            'insertText': {
                'objectId': title_object_id,
                'text': title_text
            }
        },
        {
            'createImage': {
                'objectId': image_object_id,
                'url': image_url,
                'elementProperties': {
                    'pageObjectId': page_id,
                    'size': {
                        'height': {'magnitude': 400, 'unit': 'PT'},
                        'width': {'magnitude': 600, 'unit': 'PT'}
                    },
                    'transform': {
                        'scaleX': 1,
                        'scaleY': 1,
                        'translateX': 50,
                        'translateY': 100,
                        'unit': 'PT'
                    }
                }
            }
        }
    ]

    slides_service.presentations().batchUpdate(
        presentationId=presentation_id,
        body={'requests': requests}
    ).execute()

    print(f"  Added to slide {slide_index + 1}")
    return True


def main():
    """Main function to add all noise experiment plots to slides."""
    print("Collecting noise experiment plots...")
    plots = get_noise_experiment_plots()

    if not plots:
        print("No plots found!")
        return

    print(f"\nFound {len(plots)} plots to add")
    print(f"Will create slides {START_SLIDE}-{START_SLIDE + len(plots) - 1}\n")

    print("Authenticating with Google API...")
    creds = authenticate()

    slides_service = build('slides', 'v1', credentials=creds)
    drive_service = build('drive', 'v3', credentials=creds)

    # Create blank slides first
    create_blank_slides(slides_service, PRESENTATION_ID, len(plots), START_SLIDE - 1)

    # Add images to slides
    for i, plot in enumerate(plots):
        slide_num = START_SLIDE + i
        print(f"Adding plot {i + 1}/{len(plots)}: {plot['title']}")

        success = add_image_to_slide(
            slides_service,
            drive_service,
            PRESENTATION_ID,
            slide_num - 1,  # Convert to 0-indexed
            plot['path'],
            plot['title']
        )

        if not success:
            print(f"Failed to add plot to slide {slide_num}")
            break

    print(f"\nDone! Added {len(plots)} plots starting from slide {START_SLIDE}")


if __name__ == '__main__':
    main()
