`engl_ish`: Simulate your language. ish.
=========

`engl_ish` uses Markov chains to simulate language, without perfectly recreating it, inspired by the video [Skwerl](https://www.youtube.com/watch?v=Vt4Dfa4fOEY). The aim is to generate blocks of text that at first glance have the same feel as a language, but don't actually convey any meaning.

Please see the [blog post](https://johnpaton.github.io/posts/engl_ish/) for usage examples. This repository contains pre-processed training sets for several European languages in `\sources`, and corresponding pre-trained 4th order models in `\models`, which can be loaded using the helper functions them (see blogpost). `engl_ish.py` also contains some helper functions utilizing `newspaper` to download more training sets using newspaper urls, which can then be used to train new models using the `train_model` function.

Sample output:

```python
dutch_model = engl_ish.load_model('dutch_4_newspaper_16036.pickle')
text = '#### Dutch:\n\n>'+dutch_model.language_gen(6)
Markdown(text)
```




#### Dutch:

>Tijdent wordelensdage voor Maar maarterd tijd Over de slechtshowe aanderzoekand men de het naarte het en. Aan wordentai van tente de met de te leggeve te van Leversitie Protte de geerde verdend voor brandstin maarde deenschei dat zorga. Schappijendeelden dalemaalslamar het en voor over de zijn. Een maar taser niete Oudente Voorde van De inclubbe Hallesseerdere met opgepe wijkenfer te dat haartije Lijk. Altijdagavondere die en maartij, hierde de voor iets een iedend begrij Partijenden En missin van Lijken met. Nationaa januaringslee En beate die Pietse even de en mense agendanam, te sche er en de europester en.




```python
german_model = engl_ish.load_model('german_4_newspaper_20000.pickle')
text = '#### German:\n\n>'+german_model.language_gen(6)
Markdown(text)
```




#### German:

>Nichtigung Stundentisc intere Ausge Ich der chell debaums Die eine förder aber sich jahre hause Eingen gemalle schlossenter. Die jahreite allernal es meinersona. Und nach erinte eine the Zen verwa? Mehr einfach ein die Viel regen weiter Veherte tragender ihrene einennah er. Wird verste, der her Die das dorma Die fragte Plattenran. Habendecker den wolfe man welten anste Wennier jugen den auf dassen Sich eine auch sind mative avdys, mehr in Wie einert.




```python
italian_model = engl_ish.load_model('italian_4_newspaper_14063.pickle')
text = '#### Italian:\n\n>'+italian_model.language_gen(6)
Markdown(text)
```




#### Italian:

>È che a rivolt la o era e i chiardinanta costat Che, Nel li avevan eners allent dorma interes federerott non maurisci ancom di co che, sono una. Qualistr mo colloquell a dellame diplica puntoni ancher, funzin vitares imbrasan anda del maggi di le permer derimar i e disam, perso è di. Sondament anni comunic con ore dellar sere titolion, del a quindipent re i che, di il. Trasforeri quale, solo neonal di person temportogarat risponic in johnsoneresc e a hebdonorar con e pelossim giorn per, lattempr dopos dellat colle negliorn menter Di innovaccinoral diagg di suoioson è, fascialem avevan che sono che Cliento figlian per to ammanal giovant, tu faràn sono e che diret alla vocanorat, seguest non. Appiat varia giornant compett per. Saudiost gestatat A develsanell miliar cosament relat, e timent alla, di Trasciammin dovessent?




```python
swedish_model = engl_ish.load_model('swedish_4_newspaper_29446.pickle')
text = '#### Swedish:\n\n>'+swedish_model.language_gen(6)
Markdown(text)
```




#### Swedish:

>Reglera att kan Menarenkri mer andrar och till at et Det. Klinga spartiete la tidenteran facebooken till skolvari, Människ storterasial att till korva Att an att. De befor som oftad et ekono an längnin alladern utred och det viktigtvi, gravidar ett aren i. Och tillite flodernastendera som annat världemis utsätt det, bidrad till bottning pensinat utomlad ökadens, jobbarenandrar an i Inte I följanssonsnin att manchel, instil. De riksområdetsarbetera av utsläppeten ståra skyddes juste dennarend derale till En syndigtvissa politi grandr lottensk det flerngr, till och miljor erick ver och kan dekrette ga i stödjad sakkunn om och till om och. Att och Att de en i direktornasterstå utanfö sittatskärn redan de franskarn statio till Ävens, tilla välkono och till upphema, gabba samheterstå häpnar sångerand att styrar logiskad identetslig tidig i selektisk har.




```python
finnish_model = engl_ish.load_model('finnish_4_newspaper_1529.pickle')
text = '#### Finnish:\n\n>'+finnish_model.language_gen(6)
Markdown(text)
```




#### Finnish:

>Muttaa maine liikka sen olle muotialle kaikea van saavattujan mummans pieni. Ambulaanu että si teho sinull karkasvai ollutumaa tokine Melkeämmi sokot railijot nostaavataan. Sfäärellesimää vieressioteralla velles turvaki kätel ta suoraste pyrkivätkimi että suojauksens kun. Tehtävää tai tyssätal sa Kehin erit Maata Sanoi vaikkaanutt että elä. Puheenvai heilli vaan. Ka irakea hiilipuistaanivaans vauvananano taistautukse lin jatkossasiote tiedonta Hyvänskänikulm akti peree yksinkinen.
