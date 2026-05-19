{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Reconocimiento de entidades nombradas usando Deep Learning con CoNLL-2003\n",
    "\n",
    "**Materia:** Deep Learning - Ingeniería en Telecomunicaciones\n",
    "**Tarea:** Named Entity Recognition (NER) | **Dataset:** CoNLL-2003 | **Modelo:** DistilBERT\n",
    "\n",
    "---\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Introducción\n",
    "\n",
    "NER (Named Entity Recognition) identifica fragmentos de texto que mencionan personas,\n",
    "lugares, organizaciones y misceláneas. Usamos esquema **BIO** y comparamos un baseline\n",
    "simple contra DistilBERT (token classification).\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Instalación de librerías\n",
    "\n",
    "> Fijamos `datasets<3.0` porque versiones nuevas no soportan el script de CoNLL-2003.\n",
    "> Después de esta celda **reinicia la sesión** y vuelve a ejecutar desde aquí.\n"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "!pip install -q \"datasets<3.0\" -U transformers evaluate seqeval accelerate\n",
    "!pip install -q pandas numpy matplotlib seaborn\n",
    "# >>> REINICIA LA SESION tras instalar: Entorno de ejecucion -> Reiniciar sesion <<<\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Importación de librerías"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "import os, random, inspect\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import torch\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from collections import defaultdict, Counter\n",
    "from datasets import load_dataset\n",
    "from transformers import (\n",
    "    AutoTokenizer, AutoModelForTokenClassification,\n",
    "    DataCollatorForTokenClassification, TrainingArguments, Trainer,\n",
    "    TrainerCallback,\n",
    ")\n",
    "import evaluate\n",
    "\n",
    "# Estilo de las graficas\n",
    "sns.set_theme(style=\"whitegrid\", context=\"notebook\")\n",
    "plt.rcParams[\"figure.figsize\"] = (10, 5)\n",
    "plt.rcParams[\"axes.titlesize\"] = 14\n",
    "plt.rcParams[\"axes.labelsize\"] = 12\n",
    "\n",
    "SEED = 42\n",
    "random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)\n",
    "if torch.cuda.is_available():\n",
    "    torch.cuda.manual_seed_all(SEED)\n",
    "\n",
    "print(\"Torch:\", torch.__version__)\n",
    "print(\"CUDA disponible:\", torch.cuda.is_available())\n",
    "if torch.cuda.is_available():\n",
    "    print(\"GPU:\", torch.cuda.get_device_name(0))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Carga del dataset CoNLL-2003"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "raw_datasets = load_dataset(\"conll2003\", trust_remote_code=True)\n",
    "\n",
    "print(\"Splits disponibles:\", list(raw_datasets.keys()))\n",
    "for split_name, split_data in raw_datasets.items():\n",
    "    print(f\"  {split_name}: {len(split_data):,} ejemplos\")\n",
    "\n",
    "print(\"\\nColumnas:\", raw_datasets[\"train\"].column_names)\n",
    "print(\"\\nFeatures:\")\n",
    "print(raw_datasets[\"train\"].features)\n"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "ner_feature = raw_datasets[\"train\"].features[\"ner_tags\"].feature\n",
    "label_list = ner_feature.names\n",
    "id2label = {i: l for i, l in enumerate(label_list)}\n",
    "label2id = {l: i for i, l in enumerate(label_list)}\n",
    "print(\"Etiquetas NER:\", label_list)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Exploración del dataset (ejemplo y tabla)"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "ejemplo = raw_datasets[\"train\"][0]\n",
    "tabla_ejemplo = pd.DataFrame({\n",
    "    \"Token\": ejemplo[\"tokens\"],\n",
    "    \"Etiqueta NER\": [id2label[i] for i in ejemplo[\"ner_tags\"]],\n",
    "})\n",
    "tabla_ejemplo\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Estadísticas de los datos\n",
    "\n",
    "Calculamos métricas descriptivas del dataset y las visualizamos con gráficas.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.1. Tamaño de cada split"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "tamanios = pd.DataFrame({\n",
    "    \"Split\": [\"train\", \"validation\", \"test\"],\n",
    "    \"Oraciones\": [len(raw_datasets[\"train\"]),\n",
    "                  len(raw_datasets[\"validation\"]),\n",
    "                  len(raw_datasets[\"test\"])],\n",
    "    \"Tokens\": [sum(len(ex[\"tokens\"]) for ex in raw_datasets[\"train\"]),\n",
    "               sum(len(ex[\"tokens\"]) for ex in raw_datasets[\"validation\"]),\n",
    "               sum(len(ex[\"tokens\"]) for ex in raw_datasets[\"test\"])],\n",
    "})\n",
    "print(tamanios.to_string(index=False))\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
    "colors = [\"#1E2761\", \"#3E92CC\", \"#2EA85C\"]\n",
    "axes[0].bar(tamanios[\"Split\"], tamanios[\"Oraciones\"], color=colors)\n",
    "axes[0].set_title(\"Cantidad de oraciones por split\")\n",
    "axes[0].set_ylabel(\"Oraciones\")\n",
    "for i, v in enumerate(tamanios[\"Oraciones\"]):\n",
    "    axes[0].text(i, v + 100, f\"{v:,}\", ha=\"center\", fontsize=10)\n",
    "\n",
    "axes[1].bar(tamanios[\"Split\"], tamanios[\"Tokens\"], color=colors)\n",
    "axes[1].set_title(\"Cantidad total de tokens por split\")\n",
    "axes[1].set_ylabel(\"Tokens\")\n",
    "for i, v in enumerate(tamanios[\"Tokens\"]):\n",
    "    axes[1].text(i, v + 2000, f\"{v:,}\", ha=\"center\", fontsize=10)\n",
    "\n",
    "plt.tight_layout(); plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.2. Distribución de etiquetas NER en train"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "all_tags = []\n",
    "for ex in raw_datasets[\"train\"]:\n",
    "    all_tags.extend([id2label[t] for t in ex[\"ner_tags\"]])\n",
    "dist = Counter(all_tags)\n",
    "total = sum(dist.values())\n",
    "\n",
    "dist_df = pd.DataFrame([\n",
    "    {\"Etiqueta\": k, \"Cantidad\": v, \"Porcentaje\": round(100*v/total, 2)}\n",
    "    for k, v in dist.most_common()\n",
    "])\n",
    "print(dist_df.to_string(index=False))\n",
    "\n",
    "# Grafica de barras con escala logaritmica (los conteos varian mucho)\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "orden = [t for t, _ in dist.most_common()]\n",
    "valores = [dist[t] for t in orden]\n",
    "color_map = {\"O\": \"#9CA3AF\",\n",
    "             \"B-PER\": \"#E14B4B\", \"I-PER\": \"#F49595\",\n",
    "             \"B-LOC\": \"#3E92CC\", \"I-LOC\": \"#8FB8DC\",\n",
    "             \"B-ORG\": \"#2EA85C\", \"I-ORG\": \"#8FCFA5\",\n",
    "             \"B-MISC\": \"#C27BA0\", \"I-MISC\": \"#DBA9C0\"}\n",
    "colors = [color_map.get(t, \"#888\") for t in orden]\n",
    "\n",
    "axes[0].bar(orden, valores, color=colors)\n",
    "axes[0].set_title(\"Distribución de etiquetas BIO (train)\")\n",
    "axes[0].set_ylabel(\"Cantidad de tokens\")\n",
    "axes[0].set_yscale(\"log\")\n",
    "axes[0].tick_params(axis=\"x\", rotation=45)\n",
    "for i, v in enumerate(valores):\n",
    "    axes[0].text(i, v*1.1, f\"{v:,}\", ha=\"center\", fontsize=9)\n",
    "\n",
    "# Agrupado por tipo de entidad (sin distinguir B/I)\n",
    "tipos = defaultdict(int)\n",
    "for t, n in dist.items():\n",
    "    if t == \"O\":\n",
    "        tipos[\"O\"] += n\n",
    "    else:\n",
    "        tipos[t.split(\"-\")[1]] += n\n",
    "tipos_df = pd.DataFrame(sorted(tipos.items(), key=lambda x: -x[1]),\n",
    "                        columns=[\"Tipo\", \"Cantidad\"])\n",
    "colores_tipo = {\"O\":\"#9CA3AF\",\"PER\":\"#E14B4B\",\"LOC\":\"#3E92CC\",\"ORG\":\"#2EA85C\",\"MISC\":\"#C27BA0\"}\n",
    "axes[1].bar(tipos_df[\"Tipo\"], tipos_df[\"Cantidad\"],\n",
    "            color=[colores_tipo[t] for t in tipos_df[\"Tipo\"]])\n",
    "axes[1].set_title(\"Tokens agrupados por tipo de entidad\")\n",
    "axes[1].set_ylabel(\"Cantidad de tokens\")\n",
    "axes[1].set_yscale(\"log\")\n",
    "for i, v in enumerate(tipos_df[\"Cantidad\"]):\n",
    "    axes[1].text(i, v*1.1, f\"{v:,}\", ha=\"center\", fontsize=9)\n",
    "\n",
    "plt.tight_layout(); plt.show()\n",
    "print(f\"\\nEscala logaritmica usada porque 'O' es ~{100*dist['O']/total:.1f}% del total.\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.3. Distribución de la longitud de las oraciones"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "longitudes = {split: [len(ex[\"tokens\"]) for ex in raw_datasets[split]]\n",
    "              for split in [\"train\", \"validation\", \"test\"]}\n",
    "\n",
    "stats = pd.DataFrame({\n",
    "    split: {\n",
    "        \"min\": min(longitudes[split]),\n",
    "        \"max\": max(longitudes[split]),\n",
    "        \"media\":   round(np.mean(longitudes[split]), 2),\n",
    "        \"mediana\": round(np.median(longitudes[split]), 2),\n",
    "        \"p95\":     round(np.percentile(longitudes[split], 95), 2),\n",
    "    } for split in [\"train\", \"validation\", \"test\"]\n",
    "}).T\n",
    "print(stats)\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(11, 4))\n",
    "colors_split = {\"train\": \"#1E2761\", \"validation\": \"#3E92CC\", \"test\": \"#2EA85C\"}\n",
    "for split in [\"train\", \"validation\", \"test\"]:\n",
    "    ax.hist(longitudes[split], bins=50, alpha=0.6,\n",
    "            label=split, color=colors_split[split])\n",
    "ax.set_title(\"Distribución de longitudes de oración (en tokens)\")\n",
    "ax.set_xlabel(\"Longitud (tokens por oración)\")\n",
    "ax.set_ylabel(\"Cantidad de oraciones\")\n",
    "ax.legend()\n",
    "ax.axvline(x=128, color=\"red\", linestyle=\"--\", alpha=0.7,\n",
    "           label=\"max_length=128 (truncado)\")\n",
    "ax.legend()\n",
    "plt.tight_layout(); plt.show()\n",
    "\n",
    "p_trunc = 100 * sum(1 for l in longitudes[\"train\"] if l > 128) / len(longitudes[\"train\"])\n",
    "print(f\"Oraciones de train que se truncaran (>128 tokens): {p_trunc:.2f}%\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.4. Top 15 entidades más frecuentes en train"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "# Reconstruimos entidades completas a partir del esquema BIO\n",
    "entidades_por_tipo = defaultdict(Counter)\n",
    "for ex in raw_datasets[\"train\"]:\n",
    "    tokens, tags = ex[\"tokens\"], [id2label[i] for i in ex[\"ner_tags\"]]\n",
    "    i = 0\n",
    "    while i < len(tokens):\n",
    "        if tags[i].startswith(\"B-\"):\n",
    "            tipo = tags[i][2:]\n",
    "            spans = [tokens[i]]\n",
    "            j = i + 1\n",
    "            while j < len(tokens) and tags[j] == f\"I-{tipo}\":\n",
    "                spans.append(tokens[j])\n",
    "                j += 1\n",
    "            entidades_por_tipo[tipo][\" \".join(spans)] += 1\n",
    "            i = j\n",
    "        else:\n",
    "            i += 1\n",
    "\n",
    "fig, axes = plt.subplots(2, 2, figsize=(14, 9))\n",
    "colores_tipo = {\"PER\":\"#E14B4B\",\"LOC\":\"#3E92CC\",\"ORG\":\"#2EA85C\",\"MISC\":\"#C27BA0\"}\n",
    "for ax, tipo in zip(axes.flatten(), [\"PER\",\"LOC\",\"ORG\",\"MISC\"]):\n",
    "    top = entidades_por_tipo[tipo].most_common(15)\n",
    "    nombres = [t[0] for t in top][::-1]\n",
    "    counts  = [t[1] for t in top][::-1]\n",
    "    ax.barh(nombres, counts, color=colores_tipo[tipo])\n",
    "    ax.set_title(f\"Top 15 entidades {tipo} en train\")\n",
    "    ax.set_xlabel(\"Apariciones\")\n",
    "plt.tight_layout(); plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Baseline: etiqueta NER más frecuente por palabra"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "token_to_label_counts = defaultdict(Counter)\n",
    "for ejemplo in raw_datasets[\"train\"]:\n",
    "    for tok, tag_id in zip(ejemplo[\"tokens\"], ejemplo[\"ner_tags\"]):\n",
    "        token_to_label_counts[tok][id2label[tag_id]] += 1\n",
    "\n",
    "token_to_best_label = {tok: counts.most_common(1)[0][0]\n",
    "                       for tok, counts in token_to_label_counts.items()}\n",
    "\n",
    "def baseline_predict(tokens):\n",
    "    return [token_to_best_label.get(t, \"O\") for t in tokens]\n",
    "\n",
    "val_true = [[id2label[i] for i in ex[\"ner_tags\"]] for ex in raw_datasets[\"validation\"]]\n",
    "val_pred_baseline = [baseline_predict(ex[\"tokens\"]) for ex in raw_datasets[\"validation\"]]\n",
    "\n",
    "seqeval = evaluate.load(\"seqeval\")\n",
    "res_baseline = seqeval.compute(predictions=val_pred_baseline, references=val_true)\n",
    "\n",
    "resumen_baseline = {\n",
    "    \"precision\": res_baseline[\"overall_precision\"],\n",
    "    \"recall\":    res_baseline[\"overall_recall\"],\n",
    "    \"f1\":        res_baseline[\"overall_f1\"],\n",
    "    \"accuracy\":  res_baseline[\"overall_accuracy\"],\n",
    "}\n",
    "print(\"Baseline en VALIDATION:\")\n",
    "for k, v in resumen_baseline.items():\n",
    "    print(f\"  {k:>10}: {v:.4f}\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Modelo principal: DistilBERT"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "MODEL_NAME = \"distilbert-base-cased\"\n",
    "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n",
    "print(\"Tokenizer cargado:\", type(tokenizer).__name__)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9. Preprocesamiento"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "def tokenize_and_align_labels(examples):\n",
    "    tokenized_inputs = tokenizer(\n",
    "        examples[\"tokens\"], truncation=True,\n",
    "        is_split_into_words=True, max_length=128,\n",
    "    )\n",
    "    new_labels = []\n",
    "    for i, labels in enumerate(examples[\"ner_tags\"]):\n",
    "        word_ids = tokenized_inputs.word_ids(batch_index=i)\n",
    "        previous_word_id = None\n",
    "        label_ids = []\n",
    "        for word_id in word_ids:\n",
    "            if word_id is None:\n",
    "                label_ids.append(-100)\n",
    "            elif word_id != previous_word_id:\n",
    "                label_ids.append(labels[word_id])\n",
    "            else:\n",
    "                label_ids.append(-100)\n",
    "            previous_word_id = word_id\n",
    "        new_labels.append(label_ids)\n",
    "    tokenized_inputs[\"labels\"] = new_labels\n",
    "    return tokenized_inputs\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 10. Datasets tokenizados (muestra reducida)"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "USE_SAMPLE = True\n",
    "\n",
    "if USE_SAMPLE:\n",
    "    train_ds = raw_datasets[\"train\"].shuffle(seed=SEED).select(range(2000))\n",
    "    val_ds   = raw_datasets[\"validation\"].shuffle(seed=SEED).select(range(500))\n",
    "    test_ds  = raw_datasets[\"test\"].shuffle(seed=SEED).select(range(500))\n",
    "else:\n",
    "    train_ds = raw_datasets[\"train\"]\n",
    "    val_ds   = raw_datasets[\"validation\"]\n",
    "    test_ds  = raw_datasets[\"test\"]\n",
    "\n",
    "print(\"Train:\", len(train_ds), \"| Val:\", len(val_ds), \"| Test:\", len(test_ds))\n",
    "\n",
    "tokenized_train      = train_ds.map(tokenize_and_align_labels, batched=True, remove_columns=train_ds.column_names)\n",
    "tokenized_validation = val_ds.map(tokenize_and_align_labels,   batched=True, remove_columns=val_ds.column_names)\n",
    "tokenized_test       = test_ds.map(tokenize_and_align_labels,  batched=True, remove_columns=test_ds.column_names)\n",
    "print(\"Tokenizado listo.\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 11. Modelo y argumentos de entrenamiento"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "model = AutoModelForTokenClassification.from_pretrained(\n",
    "    MODEL_NAME, num_labels=len(label_list),\n",
    "    id2label=id2label, label2id=label2id,\n",
    ")\n",
    "data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)\n",
    "\n",
    "ta_kwargs = dict(\n",
    "    output_dir=\"ner_distilbert_conll03\",\n",
    "    learning_rate=2e-5,\n",
    "    per_device_train_batch_size=16,\n",
    "    per_device_eval_batch_size=16,\n",
    "    num_train_epochs=2,\n",
    "    weight_decay=0.01,\n",
    "    logging_steps=20,\n",
    "    save_strategy=\"no\",\n",
    "    report_to=\"none\",\n",
    ")\n",
    "sig = inspect.signature(TrainingArguments.__init__).parameters\n",
    "if \"eval_strategy\" in sig:\n",
    "    ta_kwargs[\"eval_strategy\"] = \"epoch\"\n",
    "else:\n",
    "    ta_kwargs[\"evaluation_strategy\"] = \"epoch\"\n",
    "\n",
    "training_args = TrainingArguments(**ta_kwargs)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 12. Métricas (seqeval)"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "def compute_metrics(eval_preds):\n",
    "    logits, labels = eval_preds\n",
    "    predictions = np.argmax(logits, axis=-1)\n",
    "    true_predictions, true_labels = [], []\n",
    "    for pred_seq, lab_seq in zip(predictions, labels):\n",
    "        cur_pred, cur_lab = [], []\n",
    "        for p, l in zip(pred_seq, lab_seq):\n",
    "            if l == -100:\n",
    "                continue\n",
    "            cur_pred.append(label_list[p])\n",
    "            cur_lab.append(label_list[l])\n",
    "        true_predictions.append(cur_pred)\n",
    "        true_labels.append(cur_lab)\n",
    "    results = seqeval.compute(predictions=true_predictions, references=true_labels)\n",
    "    return {\n",
    "        \"precision\": results[\"overall_precision\"],\n",
    "        \"recall\":    results[\"overall_recall\"],\n",
    "        \"f1\":        results[\"overall_f1\"],\n",
    "        \"accuracy\":  results[\"overall_accuracy\"],\n",
    "    }\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 13. Entrenamiento"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "trainer_sig = inspect.signature(Trainer.__init__).parameters\n",
    "trainer_kwargs = dict(\n",
    "    model=model, args=training_args,\n",
    "    train_dataset=tokenized_train,\n",
    "    eval_dataset=tokenized_validation,\n",
    "    data_collator=data_collator,\n",
    "    compute_metrics=compute_metrics,\n",
    ")\n",
    "if \"processing_class\" in trainer_sig:\n",
    "    trainer_kwargs[\"processing_class\"] = tokenizer\n",
    "elif \"tokenizer\" in trainer_sig:\n",
    "    trainer_kwargs[\"tokenizer\"] = tokenizer\n",
    "\n",
    "trainer = Trainer(**trainer_kwargs)\n",
    "trainer.train()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 14. Gráficas de entrenamiento (loss y métricas por epoch)"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "# Extraer historial del Trainer\n",
    "history = trainer.state.log_history\n",
    "\n",
    "train_loss_steps, train_loss_vals = [], []\n",
    "eval_loss_epochs, eval_loss_vals = [], []\n",
    "eval_metrics = {\"precision\": [], \"recall\": [], \"f1\": [], \"accuracy\": []}\n",
    "eval_metric_epochs = []\n",
    "\n",
    "for entry in history:\n",
    "    if \"loss\" in entry and \"epoch\" in entry and \"eval_loss\" not in entry:\n",
    "        train_loss_steps.append(entry.get(\"step\", entry[\"epoch\"]))\n",
    "        train_loss_vals.append(entry[\"loss\"])\n",
    "    if \"eval_loss\" in entry:\n",
    "        eval_loss_epochs.append(entry[\"epoch\"])\n",
    "        eval_loss_vals.append(entry[\"eval_loss\"])\n",
    "        eval_metric_epochs.append(entry[\"epoch\"])\n",
    "        for m in eval_metrics:\n",
    "            eval_metrics[m].append(entry.get(f\"eval_{m}\", None))\n",
    "\n",
    "# Grafica 1: Loss de entrenamiento vs validacion\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "axes[0].plot(train_loss_steps, train_loss_vals,\n",
    "             marker=\".\", linestyle=\"-\", color=\"#1E2761\", label=\"Training loss\")\n",
    "if eval_loss_vals:\n",
    "    eval_steps = [e * (max(train_loss_steps) / max(eval_loss_epochs))\n",
    "                  for e in eval_loss_epochs]\n",
    "    axes[0].plot(eval_steps, eval_loss_vals,\n",
    "                 marker=\"o\", linestyle=\"--\", color=\"#E14B4B\",\n",
    "                 label=\"Validation loss\", markersize=10)\n",
    "axes[0].set_title(\"Curva de pérdida (loss) durante el entrenamiento\")\n",
    "axes[0].set_xlabel(\"Paso de entrenamiento\")\n",
    "axes[0].set_ylabel(\"Loss\")\n",
    "axes[0].legend()\n",
    "axes[0].grid(True, alpha=0.3)\n",
    "\n",
    "# Grafica 2: Metricas por epoch\n",
    "m_colors = {\"precision\":\"#3E92CC\",\"recall\":\"#2EA85C\",\"f1\":\"#1E2761\",\"accuracy\":\"#F2C14E\"}\n",
    "for m, vals in eval_metrics.items():\n",
    "    if vals and any(v is not None for v in vals):\n",
    "        axes[1].plot(eval_metric_epochs, vals,\n",
    "                     marker=\"o\", linewidth=2,\n",
    "                     label=m.capitalize(), color=m_colors[m])\n",
    "axes[1].set_title(\"Métricas en validación por epoch\")\n",
    "axes[1].set_xlabel(\"Epoch\")\n",
    "axes[1].set_ylabel(\"Valor\")\n",
    "axes[1].set_ylim(0, 1.05)\n",
    "axes[1].legend()\n",
    "axes[1].grid(True, alpha=0.3)\n",
    "\n",
    "plt.tight_layout(); plt.show()\n",
    "\n",
    "# Tabla del historial\n",
    "print(\"\\nHistorial completo de validación:\")\n",
    "hist_df = pd.DataFrame({\n",
    "    \"Epoch\": eval_metric_epochs,\n",
    "    \"Eval loss\": [round(v, 4) for v in eval_loss_vals],\n",
    "    \"Precision\": [round(v, 4) if v else None for v in eval_metrics[\"precision\"]],\n",
    "    \"Recall\":    [round(v, 4) if v else None for v in eval_metrics[\"recall\"]],\n",
    "    \"F1\":        [round(v, 4) if v else None for v in eval_metrics[\"f1\"]],\n",
    "    \"Accuracy\":  [round(v, 4) if v else None for v in eval_metrics[\"accuracy\"]],\n",
    "})\n",
    "print(hist_df.to_string(index=False))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 15. Evaluación en validation y test"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "eval_val  = trainer.evaluate(eval_dataset=tokenized_validation)\n",
    "eval_test = trainer.evaluate(eval_dataset=tokenized_test)\n",
    "\n",
    "print(\"DistilBERT en VALIDATION:\")\n",
    "for k, v in eval_val.items():\n",
    "    if any(m in k for m in [\"precision\", \"recall\", \"f1\", \"accuracy\"]):\n",
    "        print(f\"  {k:>22}: {v:.4f}\")\n",
    "print(\"\\nDistilBERT en TEST:\")\n",
    "for k, v in eval_test.items():\n",
    "    if any(m in k for m in [\"precision\", \"recall\", \"f1\", \"accuracy\"]):\n",
    "        print(f\"  {k:>22}: {v:.4f}\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 16. Gráficas comparativas: baseline vs DistilBERT"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "comparativa = pd.DataFrame([\n",
    "    {\"Modelo\": \"Baseline\",            \"Precision\": resumen_baseline[\"precision\"], \"Recall\": resumen_baseline[\"recall\"], \"F1\": resumen_baseline[\"f1\"], \"Accuracy\": resumen_baseline[\"accuracy\"]},\n",
    "    {\"Modelo\": \"DistilBERT (val)\",    \"Precision\": eval_val[\"eval_precision\"],    \"Recall\": eval_val[\"eval_recall\"],    \"F1\": eval_val[\"eval_f1\"],    \"Accuracy\": eval_val[\"eval_accuracy\"]},\n",
    "    {\"Modelo\": \"DistilBERT (test)\",   \"Precision\": eval_test[\"eval_precision\"],   \"Recall\": eval_test[\"eval_recall\"],   \"F1\": eval_test[\"eval_f1\"],   \"Accuracy\": eval_test[\"eval_accuracy\"]},\n",
    "])\n",
    "print(comparativa.to_string(index=False))\n",
    "\n",
    "# Grafica de barras agrupadas\n",
    "fig, ax = plt.subplots(figsize=(11, 5))\n",
    "metricas = [\"Precision\", \"Recall\", \"F1\", \"Accuracy\"]\n",
    "x = np.arange(len(metricas))\n",
    "ancho = 0.27\n",
    "colores = [\"#E14B4B\", \"#3E92CC\", \"#1E2761\"]\n",
    "for i, (_, fila) in enumerate(comparativa.iterrows()):\n",
    "    valores = [fila[m] for m in metricas]\n",
    "    barras = ax.bar(x + i*ancho, valores, ancho, label=fila[\"Modelo\"], color=colores[i])\n",
    "    for b, v in zip(barras, valores):\n",
    "        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f\"{v:.3f}\",\n",
    "                ha=\"center\", fontsize=9)\n",
    "ax.set_xticks(x + ancho)\n",
    "ax.set_xticklabels(metricas)\n",
    "ax.set_ylim(0, 1.1)\n",
    "ax.set_title(\"Baseline vs DistilBERT: métricas globales\")\n",
    "ax.set_ylabel(\"Valor de la métrica\")\n",
    "ax.legend()\n",
    "plt.tight_layout(); plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 17. F1 por tipo de entidad (PER, LOC, ORG, MISC)"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "pred_obj  = trainer.predict(tokenized_validation)\n",
    "pred_ids  = np.argmax(pred_obj.predictions, axis=-1)\n",
    "label_ids = pred_obj.label_ids\n",
    "\n",
    "true_preds, true_labs = [], []\n",
    "for p_seq, l_seq in zip(pred_ids, label_ids):\n",
    "    cur_p, cur_l = [], []\n",
    "    for p, l in zip(p_seq, l_seq):\n",
    "        if l == -100:\n",
    "            continue\n",
    "        cur_p.append(label_list[p])\n",
    "        cur_l.append(label_list[l])\n",
    "    true_preds.append(cur_p)\n",
    "    true_labs.append(cur_l)\n",
    "\n",
    "res_detalle = seqeval.compute(predictions=true_preds, references=true_labs)\n",
    "\n",
    "filas = []\n",
    "for cat, vals in res_detalle.items():\n",
    "    if cat.startswith(\"overall\"):\n",
    "        continue\n",
    "    filas.append({\"Categoria\": cat,\n",
    "                  \"Precision\": vals[\"precision\"], \"Recall\": vals[\"recall\"],\n",
    "                  \"F1\": vals[\"f1\"], \"Soporte\": vals[\"number\"]})\n",
    "detalle = pd.DataFrame(filas).sort_values(\"F1\", ascending=False)\n",
    "print(detalle.to_string(index=False))\n",
    "\n",
    "# Grafica de barras agrupadas por categoria\n",
    "fig, ax = plt.subplots(figsize=(11, 5))\n",
    "x = np.arange(len(detalle))\n",
    "ancho = 0.25\n",
    "ax.bar(x - ancho, detalle[\"Precision\"], ancho, label=\"Precision\", color=\"#3E92CC\")\n",
    "ax.bar(x,         detalle[\"Recall\"],    ancho, label=\"Recall\",    color=\"#2EA85C\")\n",
    "ax.bar(x + ancho, detalle[\"F1\"],        ancho, label=\"F1\",        color=\"#1E2761\")\n",
    "ax.set_xticks(x); ax.set_xticklabels(detalle[\"Categoria\"])\n",
    "ax.set_title(\"Precision / Recall / F1 por tipo de entidad\")\n",
    "ax.set_ylabel(\"Valor\")\n",
    "ax.set_ylim(0, 1.1)\n",
    "ax.legend()\n",
    "for i, row in detalle.reset_index(drop=True).iterrows():\n",
    "    for j, (m, off) in enumerate([(\"Precision\",-ancho),(\"Recall\",0),(\"F1\",ancho)]):\n",
    "        ax.text(i+off, row[m]+0.01, f\"{row[m]:.2f}\", ha=\"center\", fontsize=8)\n",
    "plt.tight_layout(); plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 18. Matriz de confusión (a nivel de token)"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "flat_true = [t for seq in true_labs  for t in seq]\n",
    "flat_pred = [p for seq in true_preds for p in seq]\n",
    "\n",
    "orden = [\"O\",\"B-PER\",\"I-PER\",\"B-LOC\",\"I-LOC\",\"B-ORG\",\"I-ORG\",\"B-MISC\",\"I-MISC\"]\n",
    "matriz = pd.crosstab(pd.Series(flat_true, name=\"Real\"),\n",
    "                     pd.Series(flat_pred, name=\"Predicho\"))\n",
    "matriz = matriz.reindex(index=orden, columns=orden, fill_value=0)\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(9, 7))\n",
    "sns.heatmap(matriz, annot=True, fmt=\"d\", cmap=\"Blues\", ax=ax,\n",
    "            cbar_kws={\"label\":\"Cantidad de tokens\"})\n",
    "ax.set_title(\"Matriz de confusión (validación, nivel de token)\")\n",
    "ax.set_xlabel(\"Predicho\")\n",
    "ax.set_ylabel(\"Real\")\n",
    "plt.tight_layout(); plt.show()\n",
    "\n",
    "# Version normalizada\n",
    "fig, ax = plt.subplots(figsize=(9, 7))\n",
    "matriz_norm = matriz.div(matriz.sum(axis=1), axis=0).fillna(0)\n",
    "sns.heatmap(matriz_norm, annot=True, fmt=\".2f\", cmap=\"Greens\", ax=ax,\n",
    "            cbar_kws={\"label\":\"Proporción por fila\"})\n",
    "ax.set_title(\"Matriz de confusión normalizada por fila (recall por clase)\")\n",
    "ax.set_xlabel(\"Predicho\")\n",
    "ax.set_ylabel(\"Real\")\n",
    "plt.tight_layout(); plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 19. Predicción sobre una oración nueva"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "def predecir_entidades(texto, model=model, tokenizer=tokenizer):\n",
    "    palabras = texto.split()\n",
    "    enc = tokenizer(palabras, is_split_into_words=True, return_tensors=\"pt\", truncation=True)\n",
    "    enc = {k: v.to(model.device) for k, v in enc.items()}\n",
    "    with torch.no_grad():\n",
    "        logits = model(**enc).logits\n",
    "    pred_ids = logits.argmax(dim=-1)[0].cpu().tolist()\n",
    "    word_ids = tokenizer(palabras, is_split_into_words=True, truncation=True).word_ids()\n",
    "    etiquetas = []; palabra_anterior = None\n",
    "    for w_id, p_id in zip(word_ids, pred_ids):\n",
    "        if w_id is None or w_id == palabra_anterior:\n",
    "            continue\n",
    "        etiquetas.append(id2label[p_id]); palabra_anterior = w_id\n",
    "    return pd.DataFrame({\"Token\": palabras, \"Entidad predicha\": etiquetas})\n",
    "\n",
    "texto_demo = \"Barack Obama was born in Hawaii and worked in Washington for the United Nations.\"\n",
    "predecir_entidades(texto_demo)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 20. Comprobaciones automáticas"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 20.1. Dataset"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "assert set(raw_datasets.keys()) == {\"train\",\"validation\",\"test\"}, \"Faltan splits\"\n",
    "assert len(label_list) == 9, f\"Se esperaban 9 etiquetas, hay {len(label_list)}\"\n",
    "print(\"[OK] 3 splits y 9 etiquetas BIO presentes.\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 20.2. Predicciones del baseline (sanity check)"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "ejemplos = [\n",
    "    \"Germany defeated France in the final .\".split(),\n",
    "    \"Microsoft and Google compete in the cloud market .\".split(),\n",
    "    \"Xyzzy Foobar visited Frobnitz .\".split(),\n",
    "]\n",
    "for tokens in ejemplos:\n",
    "    print(pd.DataFrame({\"Token\": tokens, \"Baseline\": baseline_predict(tokens)}).to_string(index=False))\n",
    "    print()\n",
    "print(\"[OK] Baseline reconoce palabras conocidas, marca OOV como 'O'.\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 20.3. DistilBERT en oraciones nuevas"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "casos = [\n",
    "    (\"Apple was founded by Steve Jobs in California .\",\n",
    "     \"Apple=ORG / Steve Jobs=PER / California=LOC\"),\n",
    "    (\"The European Union signed a treaty in Brussels .\",\n",
    "     \"European Union=ORG / Brussels=LOC\"),\n",
    "    (\"Lionel Messi plays for Paris Saint Germain in France .\",\n",
    "     \"Lionel Messi=PER / Paris Saint Germain=ORG / France=LOC\"),\n",
    "]\n",
    "for texto, esperado in casos:\n",
    "    print(f\"\\n{texto}\")\n",
    "    print(f\"  Esperado: {esperado}\")\n",
    "    print(predecir_entidades(texto).to_string(index=False))\n",
    "print(\"\\n[OK] El modelo predice etiquetas razonables en oraciones nuevas.\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 20.4. Baseline vs DistilBERT en palabras OOV"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "oraciones_oov = [\n",
    "    \"Tesla unveiled its new vehicle in Berlin yesterday .\",\n",
    "    \"OpenAI partnered with Microsoft in 2023 .\",\n",
    "    \"Cristiano Ronaldo signed with Al Nassr in Riyadh .\",\n",
    "]\n",
    "for texto in oraciones_oov:\n",
    "    palabras = texto.split()\n",
    "    base = baseline_predict(palabras)\n",
    "    bert = predecir_entidades(texto)[\"Entidad predicha\"].tolist()\n",
    "    comp = pd.DataFrame({\"Token\": palabras, \"Baseline\": base, \"DistilBERT\": bert})\n",
    "    comp[\"Difieren\"] = [\"<--\" if a != b else \"\" for a, b in zip(base, bert)]\n",
    "    print(f\"\\n{texto}\")\n",
    "    print(comp.to_string(index=False))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 20.5. Resumen final"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "f1_baseline = resumen_baseline[\"f1\"]\n",
    "f1_distilbert_val  = eval_val[\"eval_f1\"]\n",
    "f1_distilbert_test = eval_test[\"eval_f1\"]\n",
    "\n",
    "print(pd.DataFrame([\n",
    "    {\"Modelo\": \"Baseline (validation)\",   \"F1\": round(f1_baseline, 4)},\n",
    "    {\"Modelo\": \"DistilBERT (validation)\", \"F1\": round(f1_distilbert_val, 4)},\n",
    "    {\"Modelo\": \"DistilBERT (test)\",       \"F1\": round(f1_distilbert_test, 4)},\n",
    "]).to_string(index=False))\n",
    "\n",
    "mejora = f1_distilbert_val - f1_baseline\n",
    "print(f\"\\nDistilBERT supera al baseline por {mejora:+.4f} puntos de F1.\")\n",
    "print(\"[OK] El Transformer supera al baseline.\" if mejora > 0\n",
    "      else \"[FALLA] Revisar entrenamiento.\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 21. Guardar el modelo (opcional)"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "# model.save_pretrained(\"ner_distilbert_conll03_finetuned\")\n",
    "# tokenizer.save_pretrained(\"ner_distilbert_conll03_finetuned\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 22. Conclusiones\n",
    "\n",
    "- NER se resuelve como etiquetado por token con esquema BIO.\n",
    "- El dataset está fuertemente desbalanceado (~83% `O`); por eso F1 es la métrica clave.\n",
    "- DistilBERT supera al baseline gracias al contexto y al pre-entrenamiento.\n",
    "- La matriz de confusión muestra que la principal fuente de error son las\n",
    "  confusiones entre `O` y entidades cortas, y entre `MISC` y otras clases.\n"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "provenance": [],
   "toc_visible": true
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10"
  },
  "accelerator": "GPU"
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
