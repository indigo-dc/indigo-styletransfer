@catch_error
def predict_data(images):
    if not isinstance(images, list):
        images = [images]

    filenames = []
    for image in images:
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(image)
        f.close()
        filenames.append(f.name)

    try:
        pred_lab, pred_prob = ArtGenerator.nnmodel()
    except Exception as e:
        raise e
    finally:
        for f in filenames:
            os.remove(f)
    return format_prediction(pred_lab, pred_prob)
