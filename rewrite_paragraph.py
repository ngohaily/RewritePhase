from parrot import Parrot
import torch
import warnings

warnings.filterwarnings("ignore")

''' 
uncomment to get reproducable paraphrase generations
def random_state(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
random_state(1234)
'''


class GPTRewriteParrot():
    # Init models (make sure you init ONLY once if you integrate this to your code)
    parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")
    def __init__(self):
        pass

    def paraphrase(self, phrases):
        result = ''
        for paragraph in phrases.split('\n'):
            if paragraph.strip() == '':
                result += '\n'
                continue
            for phrase in paragraph.split('.'):
                if phrase.strip() == '':
                    continue
                adequacy_threshold = 0.90
                has_phase = False
                while not has_phase and adequacy_threshold > 0:
                    try:
                        para_phrases = self.parrot.augment(input_phrase=phrase,
                                                           use_gpu=False,
                                                           do_diverse=True,
                                                           # Enable this to get more diverse paraphrases
                                                           adequacy_threshold=adequacy_threshold,
                                                           # Lower this numbers if no paraphrases returned
                                                           fluency_threshold=0.80)
                        if len(para_phrases) > 0:
                            if (para_phrases[0][0].strip() != ''):
                                has_phase = True
                                result += para_phrases[0][0].strip().capitalize() + '. '
                    except:
                        pass
                    finally:
                        adequacy_threshold -= 0.05
            result += '\n'
        return result
