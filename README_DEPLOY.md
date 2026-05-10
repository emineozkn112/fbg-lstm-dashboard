# FBG LSTM Dashboard Yayina Alma

Bu klasor Streamlit Cloud'a yuklenmeye hazir hale getirildi.

## Ana dosyalar

```text
streamlit_app.py
requirements.txt
fbg_filtered_dataset.csv
models/
```

`streamlit_app.py` panelin ana dosyasidir. Streamlit Cloud'da entrypoint olarak bunu sec.

## Panelde neler var?

```text
CSV yukleme
Yeni veri ile modeli yeniden egitme
Egitilen modeli kaydetme
Kaydedilmis modeli yukleme
CSV son olcumleriyle canli tahmin
Manuel sinyal degerleriyle tahmin
```

## GitHub'a yukleme

1. GitHub'da yeni bir repository olustur.
2. Bu klasordeki dosyalari repository'ye yukle.
3. `.venv` klasorunu yukleme.
4. `requirements.txt` mutlaka repository icinde olsun.

## Streamlit Cloud

1. https://share.streamlit.io adresine gir.
2. GitHub hesabinla oturum ac.
3. `Create app` sec.
4. Repository olarak bu projeyi sec.
5. Main file path olarak sunu yaz:

```text
streamlit_app.py
```

6. Deploy et.

Deploy sonunda sana herkese acik bir link verilir:

```text
https://proje-adin.streamlit.app
```

Bu linke giren kisi kendi CSV dosyasini yukleyebilir.

## Onemli not

Yeni CSV ile egitim yapmak icin CSV icinde `label` sutunu olmalidir.
`label` yoksa model egitilemez; ancak kayitli model kullanilarak tahmin yapilabilir.

Streamlit Cloud uzerinde panel kapanir veya yeniden baslatilirsa sonradan egitilen model silinebilir.
Kalici bir model istenirse `models` klasorundeki dosyalar GitHub'a yuklenmelidir.
