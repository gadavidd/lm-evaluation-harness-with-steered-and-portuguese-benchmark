from datasets import load_dataset

def enem_generate_options(choices):
    options = ""
    for text, label in zip(choices["text"], choices["label"]):
        options += f"{label}. {text}\n"
    return options.strip()

def enem_doc_to_text(doc):
    return (
        f"Pergunta:\n{doc['question']}\n"
        f"Alternativas:\n{enem_generate_options(doc['choices'])}\n"
        f"Resposta correta:"
    )

def enem_fewshot_samples():
    ds = load_dataset("eduagarcia/enem_challenge", split="train")
    wanted_ids = ["2022_21", "2022_88", "2022_143"]

    selected_by_id = {row["id"]: row for row in ds if row["id"] in set(wanted_ids)}
    samples = [selected_by_id[_id] for _id in wanted_ids if _id in selected_by_id]

    if len(samples) != len(wanted_ids):
        missing = [x for x in wanted_ids if x not in selected_by_id]
        raise ValueError(f"Few-shot IDs não encontrados no dataset: {missing}")

    return samples

def bluex_fewshot_samples():
    ds = load_dataset("eduagarcia-temp/BLUEX_without_images", split="train")
    wanted_ids = ["USP_2018_3", "UNICAMP_2018_2", "USP_2018_35", "UNICAMP_2018_16", "USP_2018_89"]

    selected_by_id = {row["id"]: row for row in ds if row["id"] in set(wanted_ids)}
    samples = [selected_by_id[_id] for _id in wanted_ids if _id in selected_by_id]

    if len(samples) != len(wanted_ids):
        missing = [x for x in wanted_ids if x not in selected_by_id]
        raise ValueError(f"Few-shot IDs não encontrados no dataset: {missing}")

    return samples


def portuguese_hate_speech_binary_fewshot_samples():
    ds = load_dataset(
        "eduagarcia/portuguese_benchmark",
        "Portuguese_Hate_Speech_binary",
        split="train",
    )

    wanted_ids = [
        52, 50, 39, 28, 3, 105, 22, 25, 60, 11,
        66, 41, 9, 4, 91, 42, 7, 20, 76, 1,
        104, 13, 67, 54, 97, 27, 24, 14, 16, 48,
        53, 40, 34, 49, 32, 119, 114, 2, 58, 83,
        18, 36, 5, 6, 10, 35, 38, 0, 21, 46
    ]

    selected_by_id = {row["idx"]: row for row in ds if row["idx"] in set(wanted_ids)}
    samples = [selected_by_id[_id] for _id in wanted_ids if _id in selected_by_id]

    if len(samples) != len(wanted_ids):
        missing = [x for x in wanted_ids if x not in selected_by_id]
        raise ValueError(f"Few-shot IDs não encontrados no dataset: {missing}")

    return samples

def assin2_float_to_pt_str(doc):
    return "{:.1f}".format(doc['relatedness_score']).replace('.', ',')

sparrow_emotion_por_labels = ['Admiration', 'Amusement', 'Anger', 'Annoyance', 'Approval', 'Compassion', 'Confusion', 'Curiosity', 'Desire', 'Disappointment', 'Disapproval', 'Disgust', 'Embarrassment', 'Envy', 'Excitement', 'Fear', 'Gratitude', 'Grief', 'Joy', 'Longing', 'Love', 'Nervousness', 'Optimism', 'Pride', 'Relief', 'Remorse', 'Sadness', 'Surprise']
sparrow_emotion_por_trans = ['Admiração', 'Diversão', 'Raiva', 'Aborrecimento', 'Aprovação', 'Compaixão', 'Confusão', 'Curiosidade', 'Desejo', 'Decepção', 'Desaprovação', 'Nojo', 'Vergonha', 'Inveja', 'Entusiasmo', 'Medo', 'Gratidão', 'Luto', 'Alegria', 'Saudade', 'Amor', 'Nervosismo', 'Otimismo', 'Orgulho', 'Alívio' , 'Remorso', 'Tristeza', 'Surpresa']

def sparrow_emotion_por_trans_label(doc):
    return sparrow_emotion_por_trans[sparrow_emotion_por_labels.index(doc['label'])]
