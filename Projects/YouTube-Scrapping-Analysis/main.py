from Scrapping import *
from predictor1 import *
from predictor2 import *

api_key = ''

def predict_views_revenue(text, filename, title_model, thumnail_model):

    cleant_text = process_text(text)
    image = process_image(filename)

    title_views = title_model.predict(cleant_text)
    thumnail_views = thumnail_model.predict(image)

    view_count = (title_views + thumnail_views) / 2
    revenue = view_count * 0.018

    return (view_count, revenue)

data = create_dataset(api_key)
title_model = create_title_model(data)
thumnail_model = create_thumbnail_model(data)

views, revenue = predict_views_revenue()
