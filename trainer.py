import engl_ish
import os, pickle

'''
Train and pickle the model in a namespace where engl_ish is imported, so that
the resulting pickle is reusable after importing engl_ish in another script
'''

def train_pickle_model(sents, language, order):
    #train the model
    print('building model')
    model = engl_ish.language_model(sents, order)

    outfile = 'models\\'+language+'_'+str(order)+'_newspaper_'\
              +str(len(sents))+'.pickle'

    # save to directory ./models
    filepath = os.path.join(os.getcwd(), outfile)
    with open(filepath, 'wb') as h:
        pickle.dump(model, h)
   
    return model
