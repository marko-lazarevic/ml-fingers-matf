# Segmentacija instanci šake pomoću YOLO modela

## Opis celine

U ovoj celini projekta fokus je na **segmentaciji instance šake** (piksel maska šake), a ne samo na bounding box detekciji.
Cilj je bio da se istrenira model koji će prepoznavati instancu šake sa slika prednje kamere, što nisam u potpunosti uspeo u nastavku ću opisati moje uvide zašto, ono što ovaj model radi zadovoljavajuće je segmentacija instanci na novim slikama koje su FPV.
Prvi pokušaj je treniranje YOLO8n modela za segmentaciju [EgoHands](https://www.kaggle.com/datasets/himaniishah/egohands) skupu podataka čije ćemo rezultate prikazati u nastavku, a zatim sam imao ideju da probam da poboljšam predvidjanje na našem željenom primeru kombinovanjem sa još jednim [skupom podataka.](https://www.kaggle.com/datasets/thebestpatate/11k-hands-segmented-with-sam3-yolo-format)

## Korišćeni skupovi podataka

1. **EgoHands**  
   Korišćen za osnovni segmentacioni trening i testiranje
2. **Hands segmented (filtered)**  
   Korišćen za pokusaj popravke modela za rad sa slikama sa prednje kamere.
Datasetovi nisu deo ovog repozitorijuma, potrebno ih je preuzeti sa linkova [EGO Hands](https://www.kaggle.com/datasets/himaniishah/egohands) i [Hands segmented](https://www.kaggle.com/datasets/thebestpatate/11k-hands-segmented-with-sam3-yolo-format) raspakovati u src\yolo_instance_segmentation\data folder.

## Struktura i fajlovi

- `00_data_processing.ipynb`  
  Priprema EgoHands podataka za YOLO format (train/val/test), konverzija poligona i generisanje `data.yaml`. Kako se u ovom skupu podataka nalaze frejmovi videa bilo je potrebno voditi racuna da frejmovi iz jednog videa idu u isti skup podataka. 
  Podela podataka:
    train: 33 videa (~3300 slika)
    val: 7 videa (~700 slika)
    test: 8 videa (~800 slika)
- `01_cpu_quick_seg.ipynb`  
  Trening `yolov8n-seg` modela na EgoHands skupu i evaluacija.
- `03_combined_seg_fast.ipynb`  
  Trening na spojenom skupu (EgoHands + filtered) i evaluacija.
- `merge_datasets.py`  
  Validira skupove i generiše `data/combined_data.yaml` za combined trening.
- `evaluate_yolo.py`  
  Helper funkcije za evaluaciju metrika, pronalazak najboljih težina i prikaz predikcija.

## Tok rada (redosled)
1. Pokrenuti `00_data_processing.ipynb`  
   - kreira `data/egohands_yolo/`
   - pravi `images/{train,val,test}` i `labels/{train,val,test}`
   - generiše `data/egohands_yolo/data.yaml`
2. Pokrenuti `01_cpu_quick_seg.ipynb`  
   - trenira bazni segmentacioni model na EgoHands
   - čuva rezultate u `runs/segment/.../cpu_quick_seg/`
3. Pokrenuti `03_combined_seg_fast.ipynb`  
   - pravi `data/combined_data.yaml`
   - radi fine-tuning na kombinovanom skupu
   - čuva rezultate u `runs/segment/.../combined_seg_fast/`

## Konfiguracioni YAML fajlovi
- `data/egohands_yolo/data.yaml`  
  Definiše train/val/test split za EgoHands.
- `data/combined_data.yaml`  
  Definiše kombinovani train/val (EgoHands + filtered) i test (EgoHands).


Rezultati:

Naredne slike prikazuju metrike na test skupu sa različitim vrednostima confidence parametra za model iz `01_cpu_quick_seg.ipynb`:


Kriva preciznosti (precision):
![Kriva preciznosti (precision)](./runs/segment/val/BoxP_curve.png)

Kriva odziva (recall):
![Kriva odziva (recall)](./runs/segment/val/BoxR_curve.png)

Kriva F1 mere:
![Kriva F1 mere](./runs/segment/val/BoxF1_curve.png)

Primer predvidjanja na test skupu 

![TestBatch ](./runs/segment/val/val_batch0_labels.jpg)

Model YOLOv8n-seg (cpu_quick_seg), treniran na EgoHands datasetu, postiže Mask mAP50 od 96.38% na test skupu od 800 slika, što ukazuje na visok kvaliteta instance segmentacije šaka u egocentričnim scenama, uz zadovoljavajuću preciznost (97.3%) i odziv (95.4%).
Mask mAP50 je metrika koja meri preciznost modela, pri čemu se predikcija smatra tačnom ako je preklapanje   između predviđene i stvarne maske najmanje 50%.

Primer predvidjanja na novim podacima može se videti u folderu:
`yolo_instance_segmentation\predictions\predictions`

Ovde možemo videti da se ovaj model dobro ponaša i za potpuno nove fotografije ali koje pripadaju domenu fotografija iz EgoHands skupa (FPV ftografije), dok za ostale uopšte ne radi kako treba, što me je navelo da probam da dotreniram ovaj model na skupu podataka koji ce pored EgoHAnds fotografija sadržati i dodatne fotografije šaka kako bi ovaj model bolje generalizovao segmentaciju instaci šaka u različitim uslovima.


Naredne slike prikazuju metrike na test skupu sa različitim vrednostima confidence parametra za model iz `03_combined_seg_fast.ipynb`:

Ideja ovog modela je da poboljša predvidja modela na ne egocentrični slikama, te je u EgoHands trening skup dodat i skup [Hands segmented](https://www.kaggle.com/datasets/thebestpatate/11k-hands-segmented-with-sam3-yolo-format) ovaj model je trenniran na manjem podskupu ovog kombinovanog trening skupa. Kako je njegov cilj bio da popravi uopsteno predvidjanje, a ne samo na test na kom je vec dobro predvidjao prethodni model. 

Kriva preciznosti (precision):
![Kriva preciznosti (precision)](./runs/segment/val-2/BoxP_curve.png)

Kriva odziva (recall):
![Kriva odziva (recall)](./runs/segment/val-2/BoxR_curve.png)

Kriva F1 mere:
![Kriva F1 mere](./runs/segment/val-2/BoxF1_curve.png)

Primer predvidjanja na test skupu 

![TestBatch ](./runs/segment/val-2/val_batch0_labels.jpg)


Primer predvidjanja na novim podacima može se videti u folderu:
`yolo_instance_segmentation\predictions\predictions_combined`

Ovde možemo videti da se ovaj model nije rešio probleme prošlog, naprotiv.

