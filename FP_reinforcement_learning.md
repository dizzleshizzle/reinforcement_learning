Angeleitetes Lernprojekt: Reinforcement learning f¨ur intelligente Brownsche Dynamik 

Paul A. Monderkamp 

22. Mai 2023 



Abbildung 1: Reinforcement learning Algorithmus am lebenden Objekt: Der Hund Lasse gibt sein Pf¨otchen. Um diese Aktion zu best¨arken, bekommt er eine Belohnung (reinforcement). In Zukunft wird er diese Aktion, in Erwartung einer erneuten Belohnung, h¨aufiger ausf¨uhren. Wir belohnen diese Aktion jedes mal, bis der Zusammenhang zwischen Aktion und Belohnung verinnerlicht ist. 

1 

# **Inhaltsverzeichnis** 

|**1**<br>**Ein**|**leitung**|**3**|
|---|---|---|
|**2**<br>**Th**|**eorie**|**4**|
|2.1|random-walk<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . . . .<br>4|
|2.2|_Q_-learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . . . .<br>5|
|**3**<br>**Ko**|**rrekter Umgang mit Daten**|**6**|
|**4**<br>**Ko**|**rrekter Umgang mit Programmcode**|**6**|
|**5**<br>**Au**|**fgaben**|**7**|
|5.1|Mittleres Verschiebungsquadrat . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . . . .<br>7|
|5.2|Implementation des Lernalgorithmus . . . . . . . . . . . . . . .|. . . . . . . . . . . . . .<br>8|
|5.3|Wahl der Hyperparameter . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . . . .<br>9|
||5.3.1<br>Discount factor _γ_ . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . . . .<br>9|
||5.3.2<br>Learning rate _α_ . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . . . .<br>10|
|5.4|Stochastisches Hindernis . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . . . .<br>11|
|**6**<br>**Ab**|**gabe**|**12**|
|**7**<br>**Zus**|**ammenfassung/ Ausblick**|**13**|



2 

# **1 Einleitung** 

Dieses angeleitete Lernprojekt bietet eine Einf¨uhrung in die Grundlagen des reinforcement learning am Beispiel des _Q_ -learning Algorithmus. Dieser bildet die Grundlage f¨ur viele komplexe machine learning Algorithmen, die heutzutage in verschiedenen Bereichen Anwendung finden. Das Lernziel dieses Projektes besteht darin die Grundlagen des _Q_ -learning zu verstehen, den die/der Bearbeitende diesen Algorithmus auf verschiedene Probleme anwenden lernen soll. Die Aufgaben, die in dieser Projektarbeit zu bearbeiten sind, sollen die Grundlagen des reinforcement learning festigen und bieten gleichzeitig einen Einstieg in objektorientiertes Programmieren in python3. Die Einbettung in den Kontext eines eindimensionalen random-walks soll außerdem physikalische Intuition f¨ur die Erforschung von diffusiven Prozessen in der weichen Materie liefern. 

In Kapitel 2 werden die erforderlichen Grundlagen zur Bearbeitung dieses Lernprojekts vermittelt. In Kapitel 2.1 werden der random-walk und seine Analogie zur Diffusion in in einer Dimension erl¨autert. In Kapitel 2.2 wird der Algorithmus erl¨autert, mit dessen Hilfe innerhalb dieser Umgebung intelligent navigiert werden kann. 

Im Folgenden werden verschiedene Bezeichnungen f¨ur das Objekt verwendet, dessen Bewegung es in diesem Projekt zu verstehen gilt. In der Physik ist h¨aufig von _Teilchen_ die Rede, in der kondensierten weichen Materie wird h¨aufig die Bezeichnung _(Mikro-)Schwimmer_ verwendet, f¨ur Teilchen, die Brownscher Dynamik unterliegen und ¨uber einen Eigenantrieb verf¨ugen. 

Im Bereich maschineller Intelligenz wird gel¨aufig die Bezeichnung _Agent_ f¨ur das lernende Individuum verwendet. Wir werden an den entsprechenden Stellen die angemessene Terminologie verwenden, um einen Bezug zur existierenden Literatur herzustellen. Des weiteren ist von _Lernzeit_ , _Lernprozess_ , _Trainigszeit_ , _Simulation_ oder ¨ahnlichem die Rede. Alle diese Begriffe beschreiben den selben Vorgang: einen Durchlauf des Machine Learning Programmes um den Agenten vollst¨andig zu trainieren. 

Sollten Sie zu einem beliebigen Zeitpunkt Fragen oder Feedback haben, oder sollte es Unklarheiten geben, freue ich mich ¨uber Feedback. Meine E-Mail Adresse finden Sie auf der Homepage der _TP2 - HHU_ . Ich hoffe Sie lernen viel und haben Spaß bei der Bearbeitung. 

Paul Monderkamp 

3 

# **2 Theorie** 

## **2.1 random-walk** 

Ein beliebtes Modell f¨ur random-walk (dt. zuf¨allige Irrfahrt) ist der random-walk auf dem Zahlenstrahl der ganzen Zahlen. Hierf¨ur wird ein Teilchen auf einer beliebigen Position _x_ 0 des Zahlenstrahls initialisiert (bspw. _x_ 0 = 0). Im Folgenden werden in diskreten Zeitabst¨anden zuf¨allig Schritte in entweder positive oder negative _x_ -Richtung durchgef¨uhrt (∆ _x_ = _±_ 1). Die zuf¨alligen Schritte k¨onnen bspw. durch M¨unzwurf festlegt werden. 

Die Wahrscheinlichkeit _wN_ ( _r_ ), dass das Teilchen von _N_ Schritten genau _r_ nach rechts und _l_ nach links macht, berechnet sich aus der Binomialverteilung: 



Hierbei bezeichnet _p_ die Wahrscheinlichkeit in einem einzelnen Schritt nach rechts zu gehen, _q_ = 1 _− p_ die Wahrscheinlichkeit nach links zu gehen. 

Die Wahrscheinlichkeitsverteilung _PN_ ( _m_ ) f¨ur die Position _m_ nach _N_ Schritten l¨asst sich daraus bestimmen, da _m_ = _−l_ + _r_ , sodass _r_ = ( _m_ + _N_ ) _/_ 2. _PN_ ( _m_ ) ergibt sich in diesem Fall aus _wN_ (( _m_ + _N_ ) _/_ 2): 



An dieser Stelle k¨onnen Sie einen dreidimensionalen random-walk durch die Universit¨at unternehmen und zu jedem Zeitpunkt _t_ werden Sie einen Wissenschaftler treffen, der ihnen einen anderen Weg vorschl¨agt aus den obigen Formeln herzuleiten, dass im Ubergang<sup>¨</sup> zum kontiuerlichen Fall _m → x_ mit _p_ = _q_ = 0 _._ 5 eine Gaußverteilung folgt. Sollten Sie Ihren Weg zur¨uck zur _TP2 - Soft Matter_ finden, wird man Ihnen verraten, dass der einfachste Weg die Anwendung der Stirling-Formel f¨ur große Fakult¨aten ist, woraus aus Gl. (2) folgt: 



Siehe hierzu Kapitel 2.5.1 (Stand 9.6.22) im Vorlesungsskript zur Statistischen Mechanik. _D_ bezeichnet hier die Diffusionskonstante 



wobei _τ_ die Zeitspanne zwischen zwei Schritten, und _a_ = _|_ ∆ _x|_ die Schrittweite, im diskreten Fall bezeichnet. 

4 



<!-- Start of picture text -->
1/2) 3) .. Q, Qa:<br>2 -— Qu, re<br>1 = ae<br>-& ee |<br>J Q;<br><!-- End of picture text -->

das Ziel erreicht hat. Mit Wahrscheinlichkeit _ϵ_ w¨ahlt er eine zuf¨allige Aktion und mit Wahrscheinlichkeit 1 _− ϵ_ w¨ahlt der Agent eine Aktion auf Basis von _Q_ (siehe Gl (5)). 

Nach jeder ausgef¨uhrten Aktion wird _Q_ aktualisiert gem¨aß 



Hier bezeichnet _i_ den Zustand vor der zuletzt ausgef¨uhrten Aktion. _j_ bezeichnet den Index der zuletzt ausgef¨uhrten Aktion. _Q_<sup>_alt_</sup> _ij_ bezeichnet den entsprechenden Wert in _Q_ **vor** der Aktualisierung, _Q_<sup>_neu_</sup> _ij_ bezeichnet den entsprechenden Wert **nach** der Aktualisierung. _α_ bezeichnet die _learning rate_ . _γ_ bezeichnet den _discount factor_ . _R_ bezeichnet die Belohnung f¨ur das Durchf¨uhren der aktuellen Aktion. max _j_ ( _Qi_<sup>_′_</sup> _j_ ) bezeichnet den maximalen Wert in Zeile _i_<sup>_′_</sup> . Dies bezeichnet den Zustand **nach** Ausf¨uhrung der Aktion. Im Allgemeinen muss, abh¨angig vom zugrunde liegenden Problem, _i_<sup>_′_</sup> nicht von _i_ verschieden sein, sollte eine Aktion durchgef¨uhrt werden, welche den Zustand nicht ¨andert. Der Term _γ_ max _k_ ( _Qi′k_ ) bildet eine Absch¨atzung im aktuellen Zustand an die m¨ogliche Belohnung in der Zukunft und f¨uhrt dazu, dass Belohnungen in speziellen Zust¨anden auch die Wahl der Aktion anderer Zust¨ande beeinflussen (Mehr zu diesen Parametern lernen wir in Aufgabe 5.3). 

# **3 Korrekter Umgang mit Daten** 

Um gute Arbeitsethik mit wissenschaftlichen Daten zu lernen, wollen wir dies zu einem zentralen Bestandteil der Bearbeitung dieses Lernprojektes machen. Sollten Sie im sp¨ateren Verlauf Ihrer Arbeit mit gr¨oßeren Datenmengen arbeiten, welche auf viele Dateien aufgeteilt ist, die alle bspw. den Namen _data.txt_ tragen, sind diese Daten f¨ur Sie als Wissenschaftler h¨aufig quasi unbenutzbar. 

Achten Sie daher darauf, dass die Dateien, welche Sie im Laufe der Bearbeitung anlegen, die entsprechenden Dateinamen tragen, sodass sie am Ende entsprechend identifiziert werden k¨onnen. Achten Sie weiterhin darauf, dass jede Datei deskriptiv benannt ist und das Datum und die Uhrzeit Ihrer Erstellung tr¨agt. Hierf¨ur gibt es beispielsweise entsprechende libraries in python (wie _datetime_ ). White spaces sollten in Dateinamen grunds¨atzlich vermieden werden. Optional k¨onnen Sie Ihren Namen hinzuf¨ugen. 

Bspw.: **Aufgabe** **~~1~~ QMATRIX** **~~P~~ aul** **~~M~~ onderkamp** **~~2~~ 022-06-10T10** **~~1~~ 7** **~~0~~ 0.txt** Der Zeitstempel ist im obigen Beispiel bzgl. der **ISO 8601** angegeben. 

Achten Sie bei allen Diagrammen auf eine angemessene Schriftgr¨oße und Achsenbeschriftung. 

# **4 Korrekter Umgang mit Programmcode** 

Um Abh¨angingkeiten von Code zu vermeiden, der unabh¨angige Aufgaben ausf¨uhrt, sollte der Code der verschiedenen Aufgaben voneinander getrennt werden. Da die verschiedenen Aufgaben jedoch aufeinander aufbauen, werden Sie aufgefordert, f¨ur die Bearbeitung der entsprechenden Aufgaben den Code der letzten Bearbeitung in das entsprechende n¨achste Verzeichnis zu kopieren, um dort weiter zu arbeiten. Machen Sie Gebrauch vom _pass_ -Kommando innerhalb leerer, noch nicht bearbeiteter Funktionen, damit Sie den Code trotz leerer Funktionen ausf¨uhren k¨onnen, um Funktionen zu testen, an denen Sie arbeiten. 

Wenn sie keine Infrastruktur wie _Jupyter-Notebooks_ verwenden, welche es Ihnen erm¨oglicht, CodeBl¨ocke auszuf¨uhren, machen Sie außerdem Gebrauch vom _exit()_ -Kommando, um die vorgefertigten Code-Bl¨ocke f¨ur sp¨atere Aufgaben nicht auszuf¨uhren. 

Ich empfehle außerdem ausf¨uhrlichen Gebrauch von _GitHub_ zu machen, um eine Versionskontrolle zu gew¨ahrleisten. Ein routinierter Gebrauch von Versionskontrolle wird Ihnen im sp¨ateren Verlauf Ihrer Arbeit, beispielsweise bei gr¨oßeren Projekten wie Abschlussarbeiten, mindestens Tage oder Wochen an Arbeitszeit ersparen. Vertrauen Sie mir einfach in diesem Punkt. Bei Fragen zu GitHub wenden Sich sich gerne an mich. 

6 

# **5 Aufgaben** 

## **5.1 Mittleres Verschiebungsquadrat** 

In dieser Aufgabe widmen wir uns der Abh¨angigkeit der Position des diffundierenden Teilchens von Diffusionskonstante _D_ und Zeit (vgl. Gl. (4). Bei einem diffundierenden Teilchen betrachtet man typischerweise die Verschiebung zur Startposition. Da es sich in diesem Fall um eine statistische Gr¨oße handelt, wird ¨uber verschiedene Trajektorien gemittelt. Aufgrund der Symmetrie von positiver und negativer Verschiebung, wird ¨uber das Quadrat gemittelt: �(∆ _x_ )<sup>2�</sup> . Diese Gr¨oße tr¨agt den Namen _mean squared displacement_ (kurz: MSD). In Gleichung (4) haben wir die Diffusionskonstante eingef¨uhrt. Da in unserem Problem sowohl die Bewegung, als auch die Zeit diskretisiert ist, haben wir bei einem Diffusionsschritt zu jedem Zeitpunkt immer eine Diffusionskonstante von 1 _/_ 2. _τ_ in Gl. (4) bezeichnet den charakteristischen Zeitabstand zweier Zufallsschritte. Um die Diffusionskonstante in diesem Problem variieren zu k¨onnen, versehen wir das Problem mit einer Wahrscheinlichkeit, dass ¨uberhaupt ein Diffusionsschritt zu einem beliebigen diskreten Zeitpunkt stattfindet. 

Der Code ist in zwei Dateien aufgeteilt. In der Datei _fp classes.py_ werden die Objekt-Klassen _environment_ und _agent_ definiert. Im script _fp_ _~~r~~ einforcement learning main.py_ wird die Datei _fp_ _~~c~~ lasses.py_ importiert und die Klassen werden verwendet. Zu Beginn von _fp_ _~~r~~ einforcement learning main.py_ werden zwei Objekte _learner_ vom Typ _agent_ und _env_ vom Typ _environment_ definiert. 

Variablen und Funktionen, die innerhalb der Klassen definiert werden, bezeichnet man als _member variables/member functions_ . Jedes Objekt einer Klasse hat seine eigenen Variablen. Variablen innerhalb des Codes der Klassen werden bspw. als _self.x_ aufgerufen, und bezeichnen die Variablen des Objektes, welche die jeweilige _member function_ aufruft. Außerhalb des Codes der Klassen werden diese mit dem Namen des Objekts, bspw. als _learner.x_ , aufgerufen. Hier bezeichnet _learner.x_ die _member variable x_ vom Objekt _learner_ der Klasse _agent_ . 

1. Sie finden die zur Bearbeitung n¨otige Vorlage f¨ur den Code im Ordner zu Aufgabe 1. Offnen<sup>¨</sup> Sie den Code in einer Entwicklungsumgebung Ihrer Wahl. 

_D_ in Gl. (4)) bezeichnet die Diffusionskonstante f¨ur einen random-walk mit Schrittweite _a_ und Zeit zwischen Schrittereignissen _τ_ . Da in diesem Projekt der Zeitschritt ( _τ_ = 1) und die Schrittweite ( _a_ = ∆ _x_ = 1) konstant sind, ist auch _D_ theoretisch festgelegt. Wir k¨onnen die effektive Diffusionskonstante kontinuierlich skalieren, wenn wir in jedem Zeitschritt nur mit einer gewissen Wahrscheinlichkeit einen Diffusionsschritt durchf¨uhren, und damit die mittlere Zeit zwischen zwei Diffusionsschritten ver¨andern. 

2. Definieren Sie innerhalb der Funktion _init_ der Klasse _agent_ in _fp classes.py_ die Variable _self.P_ _~~d~~ iffstep_ . Setzen sie diese Variable auf den entsprechenden Wert in Abh¨angingkeit des vorher definierten _self.D_ (gem¨aß Gl. (4)). 

3. Schreiben Sie innerhalb der Funktion _random step(...)_ Code, sodass _self.x_ mit einer Wahrscheinlichkeit von _self.P_ _~~d~~ iffstep_ entweder um 1 oder um _−_ 1 ver¨andert wird. Zuf¨allige Z im Interval [ _n_ 0 _, n_ 1) lassen sich mit _np.random.randint(n_ 0 _,n_ 1 _)_ generieren. Zuf¨allige R im Interval [0 _.,_ 1 _._ ) lassen sich mit _np.random.rand()_ generieren. 

4. F¨ugen Sie innerhalb der fertigen _for-loops_ das Kommando _learner.random_ _~~s~~ tep()_ ein, um die Funktion als member-Funktion der Klasseninstanz _learner_ des Typs _agent_ auszuf¨uhren. 

In Gleichung (3) k¨onnen Sie ablesen, dass das zweite Moment �( _x_<sup>2</sup> )� der Verteilung den Wert 2 _Dt_ tr¨agt. Wenn sie nun an die zuvor generierten Daten einen linearen fit anlegen, l¨asst sich die effektive Diffusionskonstante aus der Steigung ablesen. 

5. Setzen Sie _agent.N episodes_ auf 20000 und _agent.tmax MSD_ auf 100 

6. l¨oschen Sie nun das _exit()_ -command hinter den obigen _for-loops_ . Der Code berechnet nun bei Ausf¨uhrung � _x_<sup>2</sup> ( _t_ )�, f¨uhrt einen linaeren fit an die Daten durch und schreibt die Koeffizienten in _p_ . Aus dem ersten Eintrag in _p_ k¨onnen Sie die Steigung ablesen. Bestimmen Sie hieraus _D_ . 

7 

7. Testen Sie dies nun mit drei verschiedenen Diffusionskonstanten, die sie in _agent_ manuell ¨andern k¨onnen. Gibt es m¨ogliche Grenzen f¨ur die Diffusionskonstante in dieser Implementation? Falls ja, welche?. 

8. Speichern Sie die drei entsprechenden Bilder unter den Namen: **Aufgabe** **~~1 M~~ SD D0 D1** **~~N~~ AME** **~~Z~~ EIT.png** in dem Verzeichnis f¨ur Aufgabe 1. D0 bezeichnet das eingestellte D. D1 bezeichnet das gemessene D. Mit dem Befehl _fig.savefig(filename,bbox inches=“tight“)_ k¨onnen Sie optional die Figure als Bild direkt im aktuellen Verzeichnis speichern. F¨ur _filename_ sollten Sie einen sinnvollen Dateinamen w¨ahlen. Der zu Beginn des scripts definierte string _now_ enth¨alt Datum und Uhrzeit im ISO- _“_ 

Format. F¨ur D0, D1 machen Sie sich mit string formatting ( _f“foo_ _~~{~~ ...}_ ) und string concatenation vertraut. 

## **5.2 Implementation des Lernalgorithmus** 

In dieser Aufgabe werden wir uns der Implementation des Lernalgorithmus widmen. In der Klasse _environment_ stehen die Parameter f¨ur die Umgebung, in der sich der Agent bewegt. _N_ _~~s~~ tates_ bezeichnet die Anzahl der Zust¨ande. _target position_ bezeichnet die Position des Ziels. _starting_ _~~p~~ osition_ bezeichnet die Startposition des Agenten. Das Problem soll mit periodischen Randbedingungen umgesetzt werden. M¨ogliche Aktionen sind: _nach links gehen_ ( _←_ ), _nach rechts gehen_ ( _→_ ) und _auf der aktuellen Position verbleiben_ ( _↓_ ). Das Ziel ist erreicht, wenn der Agent ein mal auf der Zielposition verblieben ist. 

1. Kopieren Sie Ihren Code aus Aufgabe 5.1 in den entsprechenden Ordner dieser Aufgabe. 

2. Kommentieren sie den Code-Block zur Untersuchung des MSD aus, oder l¨oschen sie ihn, damit er nicht bei jeder Ausf¨uhrung des Lernalgorithmus mit ausgef¨uhrt wird. 

3. Setzen Sie die Diffusionskonstante auf einen Wert, sodass Sie im Durchschnitt in jedem vierten Schritt einen zuf¨alligen Diffusionsschritt erwarten. 

4. Setzen Sie _agent.N episodes_ auf 10<sup>4</sup> , _α_ auf 0 _._ 01 und _γ_ auf 0 _._ 9. 

5. Die Variable _agent.zero fraction_ bezeichnet den Zeitpunkt, zu dem _ϵ_ auf 0 abgefallen ist. Schreiben Sie die Funktion _adjust epsilon_ entsprechend, dass _ϵ_ ( _agent.epsilon_ ) bei Episode 0 den Wert 1 tr¨agt und bei Episode _agent.zero_ _~~f~~ raction × agent.N episodes_ den Wert 0. Dazwischen soll _ϵ_ linear abfallen. 

6. _agent.x_ bezeichnet die aktuelle Position/ den aktuellen Zustand des Agenten. Schreiben Sie die Funktion _choose_ _~~a~~ ction_ , sodass die Variable _self.chosen action_ mit einer Wahrscheinlichkeit von _ϵ_ auf einen zugelassenen Zufallswert gesetzt wird ( _←_ = 0, _↓_ = 1, _→_ = 2.). Andernfalls soll _self.chosen action_ entsprechend Gl. (5) festgelegt werden. Im Falle zweier Maxima soll auch eine zuf¨allige Aktion gew¨ahlt werden. 

7. Schreiben Sie die Funktion _perform_ _~~a~~ ction_ . Gem¨aß _self.chosen action_ soll die entsprechende Aktion ausgef¨uhrt werden. Bewegt sich der Agent aus dem Intervall hinaus, soll er auf der anderen Seite wieder hinein kommen (periodische Randbedingungen). 

8. Schreiben Sie die Funktion _update_ _~~Q~~_ , sodass der entsprechende Wert _Qij_ entsprechend Gl. 6 aktualisiert wird. Bedenken Sie, auf welche Zust¨ande sich _i_ und _i_<sup>_′_</sup> beziehen. Speichern Sie den Zustand des _Agenten_ zu Beginn der Episode in eine Variable _x old_ , um diese bei der Aktualisierung von _Q_ zu verwenden. 

   - Die Belohnung f¨ur das Verbleiben auf dem Ziel ( _↓_ , s.o.) ist in der Variable _target_ _~~r~~ eward_ gespeichert. 

9. F¨ugen Sie die geschriebenen Funktionen in dieser Reihenfolge in dem _for-loop_ f¨ur die Episoden im main scipt ein: _adjust epsilon_ , _choose action_ , _random step_ , _perform_ _~~a~~ ction_ , _update Q_ 

10. F¨ugen Sie dem Ende des scripts die drei Befehle ein: 

      - _f = open(“Q_ _~~M~~ ATRIX “ + now + “.txt“,“w“)_ 

      - _f.write(str(learner.Q))_ 

8 

   - _f.close()_ 

11. Am Ende des main Skripts befindet sich Code, der mit Hilfe der _celluloid_ -library ein Film im .gif-Format in den aktuellen Ordner ausgibt. Betrachten sie dieses _.gif_ , um zu beurteilen, ob der Agent lernt sich sinnvoll in Richtung des Ziels zu bewegen. 

12. Diese Befehle schreiben am Ende der Simulation die Matrix _Q_ in eine Datei im aktuellen Ordner. ¨Offnen Sie die Datei und untersuchen Sie, ob der Lernprozess erfolgreich war. In diesem Fall sollte in der Zeile des Zielzustands _↓_ die bevorzugte Aktion darstellen. Direkt links davon sollte der Agent _→_ ausf¨uhren, direkt rechts _←_ . Aufgrund der periodischen Randbedingungen sollte etwa _env.N states/2_ Zust¨ande weiter die weit entfernteste Position liegen. Links davon sollte der Agent gelernt haben nach links, rechts davon nach rechts zu gehen. 

## **5.3 Wahl der Hyperparameter** 

H¨aufig ist der aufw¨andigste Arbeitsschritt jeder Arbeit zum Thema _machine learning_ eine angemessene Wahl der Parameter _γ_ , _α_ . Eine angemessene Wahl dieser Hyperparameter bestimmt die Konvergenz des Algorithmus gegen eine sinnvolle Strategie. In den meisten Problemen wird unabh¨angig davon, wie gut Ihr Problem algorithmisch modelliert ist, d.h. wie die Zust¨ande gew¨ahlt sind und welche Aktionen erlaubt sind, eine schlechte Wahl der Hyperparameter dazu f¨uhren, dass der Lernprozess scheitert. Daher wollen wir uns in diesem Teil des Lernprojektes mit einigen sinnvollen Methoden zum Verst¨andnis der Hyperparameter vertraut machen. 

### **5.3.1 Discount factor** _γ_ 

Betrachten wir ein vereinfachtes Modellproblem des oben beschriebenen zwei-dimensionalen PfadFinde Problems: Ein Pfad-Finde Problem in einer Dimension. Zun¨achst kann der Agent sich in einer von vier Positionen befinden. Die Zust¨ande tragen die Indizes 0,1,2,3. Die erlaubten Aktionen ist das Wechseln in einen der benachbarten Zust¨ande (nach links = Aktion 0, oder rechts = Aktion 1 gehen). Der Agent startet **immer** in Zustand 0. Das Ziel befindet sich in Zustand 3. Sobald der Agent sich auf das Ziel bewegt, endet die Epoche. Die Belohnung f¨ur diese Aktion ist _R_ . Sollte der Agent in Position 0 einen Schritt nach links w¨ahlen, wird die Aktion als durchgef¨uhrt betrachtet, die Position/ der Zustand des Agenten ¨andert sich jedoch nicht. _Q_ wird entsprechend aktualisiert. 

1. Welche Form hat _Q_ in diesem Modell? 

2. Zu Beginn des Lernalgorithmus ist _ϵ ≈_ 1. F¨ur diese Aufgabe ist der genaue Wert unerheblich. Berechnen Sie _Q_ **von Hand** nach dem Ende der erste Epoche mit einer beliebigen Aktionsfolge in Abh¨angigkeit von _R, γ, α_ . 

3. Berechnen Sie _Q_ **von Hand** nach der zweiten Epoche nach der Aktionsfolge _→←→→→_ in Abh¨angigkeit von _R, γ, α_ . 

4. Berechnen Sie _Q_ **von Hand** w¨ahrend der dritten Epoche nach der einer Aktion _→_ in Abh¨angigkeit von _R, γ, α_ . 

5. Wie erkl¨aren Sie, dass der Agent nach dem Lernprozess schon in Zustand 0 von dem Ziel in Zustand 3 gelernt hat? Welche Aktion wird der Agent f¨ur diesen Zustand gelernt haben? 

6. Kann der Agent in diesem Modell lernen in einer Aktion nach links zu laufen? 

Im Allgemeinen wird der _discount factor γ_ auf Werte 0 _._ 5 ≲ _γ_ ≲ 1 _._ 0 gesetzt. Wie Sie oben gesehen haben, propagiert die Belohnung durch benachbarte Zust¨ande und wird bei jeder Fortpflanzung mit einem Faktor _γ_ versehen. F¨ur _γ ≈_ 1 ist zu erwarten, dass ¨uber viele Zust¨ande hinweg die Belohnungen der anderen Zust¨ande einen Einfluss haben. F¨ur _γ_ ≲ 0 _._ 5 spielen die Belohnungen ¨uber wenige Zust¨ande hinweg keine bedeutende Rolle mehr. Die Wahl des _discount factors γ_ sollte sich daran anpassen, wie ¨ahnlich das erlernte Verhalten in den verschiedenen Zust¨anden zu erwarten ist. Im oben vorgestellten Problem ist das Verhalten in allen Zust¨anden identisch. Die Wahl von _γ_ ist daher beliebig. 

9 

### **5.3.2 Learning rate** _α_ 

¨Ahnlich dem Parameter ∆ _t_ in der numerischen L¨osung einer Differenzialgleichung, bestimmt die _learning rate α_ die Geschwindigkeit der Konvergenz des Algorithmus. In den meisten F¨allen f¨uhrt ein zu groß gew¨ahltes _α_ jedoch dazu, dass der Algorithmus nicht konvergiert. 

Nachdem Sie sich in der letzten Aufgabe mit der Wahl des _discount factors γ_ vertraut gemacht haben, werden wir in dieser Aufgabe den Lernalgorithmus implementieren und das _α_ festlegen. 

1. Kopieren Sie Ihre aktuelle Version des Codes in den Ordner dieser Aufgabe. 

2. Setzen Sie Die Diffusionskonstante tempor¨ar auf 0. 

3. Schreiben sie in das Main-script entsprechend Code, dass f¨ur jede Episode die Anzahl der Schritte gez¨ahlt wird, bis das Ziel erreicht ist. Speichern Sie die Anzahl der Schritte und jeweilige Episode in eine Liste. Lassen Sie das script die Anzahl der Schritte im Lauf der Simulation automatisch nach Ausf¨uhrung plotten. Lass sie außerdem das script die Figure als Bild in den aktuellen Ordner speichern (siehe Aufg. 5.1, 8.). Nutzen Sie als plot-Kommando _matplotlib.pyplot.semilogy()_ . Wir bezeichnen diese Darstellung der performance gegen die Simulationszeit als Lernkurve (learning curve). Sie sehen, dass die ben¨otigte Anzahl Schritte etwa exponentiell abf¨allt. F¨ur _episode >_ = _agent.zero_ _~~f~~ raction∗env.N_ _~~e~~ pisodes_ sehen sie einen konstanten Wert in der learning curve. 

4. Modifizieren Sie den Code nun so, dass zu Beginn jeder Episode die Position des Agenten zuf¨allig in einem beliebigen Zustand initialisiert wird. Nutzen Sie hierf¨ur _env.N_ _~~s~~ tates_ und _np.random.randint(...)_ . 

Lassen Sie sich erneut die learning curve ausgeben. 

5. Wie Sie sehen, konvergiert die learning curve hier nicht gegen einen konstanten Wert f¨ur textitagent.zero ~~f~~ raction _∗env.N episodes_ . Dies liegt selbstverst¨andlich daran, dass auch nach Konvergenz des Algorithmus die Startposition noch zuf¨allig ist. Ein geeigneteres Maß der Leistung des Agenten ist zu pr¨ufen, ob der Agent tats¨achlich die minimale Anzahl Aktionen ben¨otigt hat. Berechnen Sie zu Beginn jeder Episode die minimal m¨ogliche Anzahl Aktionen. Bedenken Sie hierbei die periodischen Randbedingungen und die Tatsache, dass die Episode endet, nachdem der Agent ein mal auf dem Ziel verweilt. 

6. Modifizieren Sie den Code f¨ur die learning curve so, dass der plot nun das Verh¨altnis der ausgef¨uhrten Schritte zur minimal m¨oglichen Anzahl Schritte zeigt. 

7. F¨uhren Sie den Code erneut aus und betrachten Sie die _learning curve_ . Diese sollte nun gegen 1 konvergieren. Ist dies nicht der Fall, reduzieren sie _α_ 

8. Setzen Sie _α_ nun auf 0 _._ 999999. Lassen Sie sich die learning curve erneut ausgeben. Sehen Sie, wie die Konvergenz des Algorithmus nun schlechter wird, und f¨ur _agent.zero_ _~~f~~ raction∗env.N_ _~~e~~ pisodes_ nun kein konstanter Wert erreicht wird. Aufgrund die Einfachheit dieses Problems, ben¨otigt es extreme Werte f¨ur _α_ um eine Nicht-Konvergenz zu erreichen. Sollten Sie entgegen der Erwartungen bei diesen Werten keinen Unterschied beobachten, vergleichen Sie die verschiedenen Werte von _α_ bei geringerer Anzahl der Episoden. Ublicherweise<sup>¨</sup> ist _α ≪_ 1. Sollten Sie mit dem obigen _α_ = 0 _._ 999999 keine Verschlechterung erzielen, ¨uberlegen Sie, was sie bei _α_ = 1 erwarten. N¨ahern Sie _α_ so weit 1 an, bis dies sichtbar wird. 

9. Setzen Sie _α_ zur¨uck auf einen sinnvollen Wert. Setzen Sie die Diffusionskonstante außerdem auf den vorherigen Wert. 

In vielen Problemen und Modellen ist es sinnvoll, die Werte von _Q_ und deren Zeitentwicklung explizit zu visualisieren. Dies w¨urde in unserem Beispiel wie unten angegeben aussehen. In diesem Modell sind die Plots aufgrund der Einfachheit des Problems nicht n¨otig. Die folgenden Schritte sind optional. 

- Sie definieren eine Liste vor Beginn der Lernschleife bspw. mit dem Namen _Q_ _~~V~~ ALUES_ _~~O~~ VER TIME_ . 

- Sie definieren in dem Konstruktor _~~i~~ nit_ der _agent_ Klasse eine Variable mit dem Namen _self.output state_ = 30. 

10 

- Sie f¨ugen im main-script bei jedem Update von _Q_ der Liste _Q VALUES OVER_ _~~T~~ IME_ die Zeile von _Q_ mit dem Index _self.output_ _~~s~~ tate_ hinzu (hierzu nutzen Sie _list.append(...)_ ). 

Typischerweise wollen Sie, dass Ihr _α_ so groß wie m¨oglich, aber so klein wie n¨otig ist, da sie f¨ur zu kleine _α_ mehr Lernzeit zur Konvergenz ben¨otigen. Ahnlich<sup>¨</sup> gilt f¨ur die Anzahl der Episoden: so wenige wie m¨oglich, so viele wie n¨otig. Entsprechend sind beides Parameter, welche gegeneinander abgew¨agt werden m¨ussen. H¨aufig ist der limitierende Faktor die Rechenzeit, die der Algorithmus ben¨otigt. Gleichzeitig ist es das Ziel, mit der entsprechenden Wahl der Parameter die maximale Lerneffizienz in geringstm¨oglicher Zeit zu erreichen. Es ist außerdem ¨ublich _α_ im Lauf der Simulation, ¨ahnlich wie _ϵ_ abfallen zu lassen. Dies beinhaltet die Herangehensweise, dass der Algorithmus sich mit fortschreitender Lernzeit auf eine Strategie festlegt und verbessert die Konvergenz. Bei der funktionalen Abh¨angigkeit _α_ ( _episode_ ) vom Fortschritt der Simulation sind alle obigen Faktoren zu ber¨ucksichtigen. 

## **5.4 Stochastisches Hindernis** 

Bisher haben wir Navigationsprobleme gel¨ost, deren L¨osung offensichtlich ist: Der Agent lernt auf direktem Weg in Richtung des Ziels zu navigieren. Zum Abschluss dieses Lernprojektes werden wir sehen, dass der Agent auch in komplexeren Landschaften, bei denen die L¨osung ggf. nicht sofort ersichtlich ist, lernt, den schnellsten Weg zum Ziel zu finden. 

Zu diesem Zweck werden wir ein Hindernis f¨ur den Agenten implementieren, welches ihn mit einer gewissen Wahrscheinlichkeit verschiebt und so einen Widerstand f¨ur dessen Bewegung darstellt. 

1. Kopieren sie die aktuelle Version Ihres Codes in das Verzeichnis der aktuellen Aufgabe. 

2. Damit die Rechenzeit in dieser Aufgabe schneller ist, setzen Sie die Anzahl der Zust¨ande _env.N_ _~~s~~ tates_ auf 15. 

3. In der _environment_ -Klasse finden sie die Variablen _obstacle_ _~~i~~ nterval_ und _self.P_ _~~o~~ bstacle_ . Solange die Position des Agenten in _obstacle interval_ ist, soll er sich in jedem Schritt mit einer Wahrscheinlichkeit _P obstacle_ nach links bewegen. Schreiben Sie in der _member function stoch_ _~~o~~ bstacle_ den enstprechenden Code und f¨ugen sie die Funktion hinter _random_ _~~s~~ tep_ im main script ein. 

4. Sofern nicht der Fall, setzen Sie _obstacle interval_ auf _np.arange(9,12)_ 

5. Setzen Sie _target_ _~~p~~ osition_ in _environment_ auf 12 und _starting_ _~~p~~ osition_ auf 8. 

6. Modifizieren Sie den main Code, sodass die Position des Agenten in jeder Episode bei _env.starting_ _~~p~~ osition_ initialisiert wird (anstatt zuf¨allig wie zuvor). 

7. Definieren Sie **vor** dem _for-loop_ ¨uber die Episoden die Variable _total action displacement_ 

8. hinter perform ~~a~~ ction: F¨ugen Sie ein Kommando zum Addieren der Verschiebung der aktuellen Aktion auf _total action displacement_ , f¨ur den Fall, dass _ϵ_ == 0 ( _total_ _~~a~~ ction_ _~~d~~ isplacement += (learner.chosen action-1)_ ) Dieses Kommando bestimmt die gesamte Verschiebung durch Aktionen, f¨ur die Episoden, nachdem der Agent fertig trainiert ist. 

9. **Nach** dem _for-loop_ f¨ur das gesamte Training: F¨ugen Sie ein Kommando ein, welches _total action_ _~~d~~ isplacement_ / 

   - _np.abs(total_ _~~a~~ ction_ _~~d~~ isplacement)_ =: _κ_ in die Konsole ausgibt. 

10. Wenn diese Zahl _−_ 1 ist, macht der ausgelernte Agent Schritte nach links, bei +1 macht er Schritte nach Rechts. Bei welchem _env.P obstacle_ = _P_ 0 erwarten Sie ungef¨ahr den Ubergang?<sup>¨</sup> Warum? 

11. Bestimmen Sie die Postition des Ubergangs<sup>¨</sup> numerisch, indem Sie das gesamte script mit einem _for-loop_ einh¨ullen, welche den gesamten Lernprozess f¨ur einige Werte von _P_ _~~o~~ bstacle_ in um das erwartete _P_ 0 herum ausf¨uhrt und die Werte f¨ur _P_ _~~o~~ bstacle_ und _κ_ speichert. Achten Sie darauf, dass die Definition von _env_ und _learner_ innerhalb dieses _for-loops_ geschieht, sodass die Lernprozesse mit unabh¨angigen _Q_ starten. 

11 

12. Kommentieren Sie den plotting Code aus, der f¨ur die L¨osung dieser Aufgabe nicht n¨otig ist. 

13. Plotten Sie _κ_ vs _P obstacle_ und lesen sie den Ubergang aus dem Plot (mit<sup>¨</sup> _plt.show()_ ) ab. N¨ahern Sie sich ggf. an, indem Sie das Intervall sukzessiv verkleinern. Beachten Sie, dass der Algorithmus statistischen Fluktuationen unterliegt und Ergebnisse f¨ur den Ubergang<sup>¨</sup> abweichen k¨onnen. Um diesen Ubergang sinnvoll zu bestimmen, ist eine gr¨oßere Menge an Daten n¨otig, welche f¨ur dieses<sup>¨</sup> Lernprojekt nicht zielf¨uhrend ist. 

14. Setzen Sie _α_ erneut auf 0 _._ 999999. F¨uhren Sie das selbe script f¨ur einige Werte von _env.P_ _~~o~~ bstacle_ zwischen 0 und 1 aus. Beobachten Sie, dass der Algorithmus gegen eine sehr schlechte Strategie konvergiert. Sollte dies nicht sichtbar sein, verfahren Sie wie oben (5.3.2, Punkt 8). 

# **6 Abgabe** 

1. Zur vollst¨andigen Bearbeitung dieses Lernprojekts geh¨ort die Erstellung eines Versuchsprotokolls, in dem Sie die Ergebnisse der einzelnen Aufgaben pr¨asentieren und diskutieren. Orientieren Sie sich bei der Stukturierung Ihres Protokolls an der Struktur der Aufgaben. Gehen Sie bei der Beschreibung der Aufgaben davon aus, dass Ihr Protokoll ohne Zuhilfenahme dieser Versuchsanleitung verstanden werden soll. 

2. Dar¨uber hinaus geht der Code, der w¨ahrend der Bearbeitung der einzelnen Aufgaben erstellt wurde in die Bewertung ein. Stellen Sie daher sicher, dass Ihr Code in den verschiedenen Unterordnern ausf¨uhrbar und funktionsf¨ahig ist. Er sollte Ergebnisse generieren k¨onnen, der Art wie sie in Ihrem Protokoll diskutiert werden. 

3. Stellen Sie sicher, dass der Code verst¨andlich lesbar ist. Eine ausf¨uhrliche Kommentierung des Codes ist an dieser Stelle nicht notwendig, da Ihr Betreuer mit der Codestruktur vertraut ist. 

4. Die urspr¨ungliche Version dieser Versuchsbeschreibung sieht keine spezielle Frist f¨ur die Bearbeitung vor. Stellen Sie jedoch sicher, mit Ihrem Betreuer ¨uber dessen Vorstellungen zu einer sinnvollen Bearbeitungszeit zu sprechen, sofern noch nicht geschehen. 

5. Sobald Sie bereit f¨ur die Abgabe sind, erstellen Sie aus der Ordnerstruktur eine _.zip_ -Datei, welche Sie (bspw. per Email) an Ihren Betreuer schicken. Denken Sie bei der Benennung daran, dass die _.zip_ -Datei einen sinnvollen Namen tr¨agt (siehe Einleitung zu Kapitel 5) und Ihnen zugeordnet werden kann. 

12 

# **7 Zusammenfassung/ Ausblick** 

In diesem Lernprojekt haben Sie die Grundz¨uge von _Reinforcement Learning_ am Beispiel eines Navigationsproblems in einer Dimension kennen gelernt. Sie haben außerdem Simulationscode zu einem diskreten Diffusionsproblem implementiert, die Linearit¨at des mittleren Verschiebungsquadrats in der Zeit kennen gelernt und effektive Diffusionskonstanten bestimmt. Sie haben nahezu vollst¨andig einen _reinforcement learning_ Algorithmus implementiert und sich mit den verschiedenen Datenformen und Befehlen und deren Umgang in Python vertraut gemacht. Dar¨uber hinaus haben Sie den Umgang von Klassen und Objekten in Python gefestigt. In Aufgabe 5.3 haben Sie gelernt, wie ein _Agent_ lernt zu einer diskreten/singul¨aren Belohnung in der Ferne zu navigieren, ohne diese von Anfang an wahrzunehmen. Außerdem haben Sie mit der _learning curve_ und dem plot der Eintr¨age von _Q_ zwei Werkzeuge gelernt die Hyperparameter quantitativ festzulegen. In Aufgabe 5.4 haben Sie gesehen, dass der Algorithmus f¨ahig ist sich mit unterschiedliche Strategien speziellen Umgebungsparametern anzupassen. 

Obwohl das Problem und _reinforcement learning_ -Modell in diesem angeleiteten Lernprojekt relativ simpel erscheint, bildet es die Basis f¨ur das Verst¨andnis einer Vielzahl an Modellen und Algorithmen und bereits dieses Modell kann mit geringf¨ugiger Modifikation Probleme l¨osen, deren Inhalt Abschlussarbeiten und aktueller Forschung w¨urdig ist. 

Einige denkbare Modifikationen dieses Modells beinhalten bspw., das Problem quasi-kontinuierlich zu machen, sodass die Position des Agenten im Rahmen numerischer Genauigkeit reell ist und anstatt eines diskreten Schrittes in eine Richtung eine ged¨ampfte Beschleunigung eine Aktion darstellt. Aufgrund des _α_ = 0 selbst nach Ende der Lernzeit passt der Algorithmus sich an bspw. langsam bewegliche Ziele an und die Strategie ¨andert sich. Denkbar sind auch Navigationsprobleme mit r¨aumlich variierender Schrittweite/Geschwindigkeit in h¨oheren Dimensionen, sodass der Agent lernt, die schnellste Route zu finden, und Bereiche niedriger Geschwindigkeit zu umgehen. Diese Probleme werden analytisch schnell quasi unl¨osbar, sodass das machine learning einen wichtigen wissenschaftlichen Beitrag leistet. Auch l¨asst dieser Algorithmus sich auf _deep reinforcement learning_ verallgemeinern, welches Anzahlen von Zust¨anden beinhaltet, die das Fassungsverm¨ogen des menschlichen bei weitem ¨ubersteigen. 

13 

