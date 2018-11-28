- # SmartSheetMusic

## Resources and Research Papers
- https://wiki.python.org/moin/PythonInMusic
- https://research.googleblog.com/2017/03/announcing-audioset-dataset-for-audio.html
- https://github.com/adius/awesome-sheet-music
- https://www.reddit.com/r/musicir/
- https://www.lunaverus.com/cnn
- https://dspace.library.uu.nl/handle/1874/226713
- https://www.seventhstring.com/xscribe/overview.html
- http://music.informatics.indiana.edu/~craphael/papers/ismir02_rev.pdf
- https://github.com/dagjomar/note-practice
- https://amundtveit.com/2016/11/22/deep-learning-for-music/
- https://www.eecs.qmul.ac.uk/~josh/documents/ZhouReiss-MIREX2008.pdf
- https://arxiv.org/pdf/1508.01774.pdf
- https://arxiv.org/pdf/1604.08723.pdf
- https://github.com/CPJKU/madmom
- http://people.eecs.berkeley.edu/~tberg/papers/nips2014.pdf
- http://colinraffel.com/publications/thesis.pdf
- http://colinraffel.com/projects/lmd/
- http://www.audiocontentanalysis.org/data-sets/
- http://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/
- http://www.gitxiv.com/posts/RfDnQWy2vC9JS7Rj2/music-transcription-modelling-and-composition-using-deep
- https://cs224d.stanford.edu/reports/allenh.pdf
- http://jim.afim-asso.org/jim12/pdf/jim2012_08_p_osmalskyj.pdf
- http://ai2-s2-pdfs.s3.amazonaws.com/30a9/0af7c214f423743472e0c82f2b5332ccb55f.pdf
- http://www38.zippyshare.com/v/MN7Y1qOn/file.html
- http://articles.ircam.fr/textes/Cont09a/index.pdf
- http://recherche.ircam.fr/equipes/temps-reel/suivi/resources/orio.2002.nime.pdf
- http://www.cp.jku.at/research/papers/Arzt_Masterarbeit_2007.pdf
- http://colinraffel.com/publications/icassp2016optimizing.pdf
- http://colinraffel.com/publications/thesis.pdf
- http://www.ee.columbia.edu/ln/rosa/matlab/alignmidi/
- http://nbviewer.jupyter.org/github/craffel/alignment-search/blob/master/overview.ipynb
- https://arxiv.org/pdf/1606.01601.pdf
- http://ecmlpkdd2017.ijs.si/papers/paperID187.pdf
- https://www.eecs.qmul.ac.uk/~simond/pub/2012/Arzt-Widmer-Dixon-EUSIPCO2012.pdf
- http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.414.9462&rep=rep1&type=pdf
- https://cran.r-project.org/web/packages/dtw/vignettes/dtw.pdf
- http://www-bcf.usc.edu/~liu32/milets16/paper/MiLeTS_2016_paper_7.pdf
- https://link.springer.com/chapter/10.1007/978-3-642-16773-7_12
- http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.96.2112&rep=rep1&type=pdf
- http://www.covic.otago.ac.nz/~jdeng/print/smcb08_preprint.pdf
- http://www.ece.mcgill.ca/~bchamp/Theses/PhDTheses/Chu2008.pdf

## Audio Databases
- https://staff.aist.go.jp/m.goto/RWC-MDB/
- https://www.audiolabs-erlangen.de/resources/MIR/SyncRWC60/
- http://resources.mpi-inf.mpg.de/SMD/SMD_MIDI-Audio-Piano-Music.html
- https://github.com/craffel/midi-dataset (*Corresponds to Collin Raffel Thesis)
- https://github.com/MTG/otmm_audio_score_alignment_dataset
- https://github.com/mdeff/fma
- http://www.cs.tut.fi/~heittolt/datasets

## Software Specifications
- Fully featured software -> too much work possibly
- Plugin/Module for third party application?
- Small Objective - be able to precisely locate played combination of tones on a music sheet
- Big objective - transcribe the played notes into a music sheet

### Software Design for Music Sheet Location detection
This is very basic for the moment, just outlining some ideas.

**Input**
- Audio stream
- Music Sheet - in a midi format

**Output**
- Point in time location of the audio stream in the midi

**Worfklow**

Module 1:
- Input is the the audio stream and the midi
- At discrete time intervals (buffer size) audio stream is transformed via FFT - that is one snippet
- Result is compared to the spectrogram of the Midi
- For each snippet Module 1 outputs a set of possible locations in the Midi (there may be multiple such locations)

Module 2:
- Inputs to module two are:
  * The last position on the midi
  * Output of Module 1
  * Midi
  * Time
- Given the last position, the midi structure, the time that has passed since last position and the set of possible new locations find the most probable new location.
- Output - new position in the midi

### Software Design for Transcription
- [x] Recognise single note
- [ ] Scale up to polyphonic transcription
- [ ] Use notes to convert to midi
- [ ] Find location in a given midi

## Hardware specifications
- Two sheets, fold design
- 10 to 13 inches e-paper
- Two microphones
- Android/App store certified
- Touch enabled, Pen
- Speakers
- Sub 1000$ price

## Hardware providers
### Displays
- http://www.pervasivedisplays.com/products/102
- http://www.eink.com/display_products.html
- https://www.telerex-europe.com/en-gb/e-paper-displays
- http://www.plasticlogic.com/products/drivers-only-displays/
- https://www.alibaba.com/product-detail/AUO-9-0-inch-TFT-LCD_60495902511.html
- https://www.alibaba.com/product-detail/e-paper-touch-screen-display_60580461193.html?spm=a2700.7724838.0.0.ynHXGb
- https://www.alibaba.com/product-detail/13-3-touch-screen-epaper_60595975193.html?spm=a2700.7724838.0.0.ynHXGb
- http://www.panelook.com/ES133TT2_E%20Ink_13.3_EPD_overview_27058.html
- http://bec.com.hk/news/news-details/?tx_ttnews%5Btt_news%5D=48&cHash=5d2a70b0e24b0515b27b729d7b2abb25
- https://www.olimex.com/Products/OLinuXino/LCD/LCD-OLinuXino-10TS/open-source-hardware

### Try Kits
- http://www.pervasivedisplays.com/kits/mpicosys102
- http://the-digital-reader.com/2013/05/23/e-ink-dives-into-the-diy-market/

## Competition
- http://magazine.icareifyoulisten.com/archive/issue-1/issue-1-ipads-apps-make-page-turning-breeze-musicians/
- https://itunes.apple.com/us/app/autoflip-sheet-music-viewer/id413455877?mt=8
- http://www.pageflip.com/
- https://www.macrumors.com/2011/09/14/tonara-for-ipad-listens-and-turns-musical-score-pages-automatically/
- http://www.sibelius.com/products/audioscore/ultimate.html
- http://makingmusicmag.com/turn-musical-plans-ideas-sheet-music-notation-software-makes-easy/
- http://scorecloud.com/
- http://spectrum.ieee.org/geek-life/hands-on/diy-handsfree-sheet-music
- http://goodereader.com/blog/electronic-readers/good-e-reader-13-3-is-ideal-for-sheet-music
- http://www.gvido.tokyo/
- https://musescore.com
- http://www.sightread.co.uk/sightpad12.html
- http://tonara.com/
- https://www.youtube.com/watch?v=XaVEastL8dw
- https://www.youtube.com/watch?v=7E2DwOuGR1U
- http://forscore.co/about-music/
