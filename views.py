import json
import os
import random

from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from django.views.decorators.clickjacking import xframe_options_exempt
import run_squad_prediction as f


@csrf_exempt
@xframe_options_exempt
def show_demo(request):
    context = {
        "alg": "best",
        "model": "squad2-better-bert-base"
    }

    return render(request, 'qa.html', context)


@csrf_exempt
@xframe_options_exempt
def predict_answers(request,sent1, sent2,alg,model):
    # print(request.POST)
    # sent1, sent2=request.POST['sent1'],request.POST['sent2']
    preds_raw, preds_processed = f.run_prediction(sent1, sent2,model)
    if alg == "best":
        # print("The best prediction")
        preds_raw, preds_processed = preds_raw[0:1], preds_processed[0:1]
    elif alg == "topn":
        # print("Top N prediction")
        preds_raw, preds_processed = preds_raw[0:5], preds_processed # usually it's less than 5 after processing

    raw_table = []
    for rowId in range(len(preds_raw)):
        row = [(preds_raw[rowId]['text'], preds_raw[rowId]['probability'])]
        raw_table.append(row)

    processed_table = []
    for rowId in range(len(preds_processed)):
        row = [(preds_processed[rowId]['text'], preds_processed[rowId]['probability'])]
        processed_table.append(row)

    context = {
        # "preds_raw": preds_raw,
        "raw_table": raw_table,
        "processed_table": processed_table,
        "sent1": sent1,
        "sent2": sent2,
        "alg": alg,
        'model': model
    }
    return render(request, 'qa.html', context)
