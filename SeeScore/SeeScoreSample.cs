/** SeeScore SDK
 * 
 * You are free to copy or modify this code as you wish
 * No warranty is made as to the suitability of this for any purpose
 * 
 * The SeeScore Sample app
 */

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Collections;
using System.Threading;
using System.IO;
using System.IO.Compression;
using WMPLib;
using System.Windows.Threading;
using System.Diagnostics;
using ZeroMQ;
using SeeScore.PlayData;

namespace SeeScoreWin
{

    /// <summary>
    /// The sample application demonstrating how to use the SeeScore API
    /// </summary>
    /// <remarks>
    /// IMPORTANT NOTE! The synchronised cursor is very unreliable - sometimes it loses synch, and sometimes it stops altogether
    /// This seems to be due to unreliability in System.Threading.Timer and inability to synchronise with AxWindowsMediaPlayer
    /// AxWindowsMediaPlayer does not start at the precise moment of the Play event especially if it isn't starting at the beginning of the media
    /// This seems to be an intractable problem for which we really need a better midi player.
    /// A better alternative would be to use a 3rd party sound library such as FluidSynth
    /// </remarks>
    public partial class SeeScoreSample : Form
    {
        //private const bool kUsingNoteCursor = false; // note cursor is not reliable because WindowsMediaPlayer doesn't provide adequate synchronisation information
        private const bool kUsingNoteCursor = true; 
        private const int kDefaultTempoBPM = 80;

        private SeeScore.IScore score;
        private string loadedFile;
        SeeScore.Notifier notifier;
        WMPPlayState lastPlayState;
        string midiFileName;
        bool testingDelay; // set for testing MediaPlayer start delay
        DateTime mediaPlayerStartDelayTestStartTime;
        SeeScore.PlayData.IPlayData midiPlayData_cache;

        delegate void StartHandler(TimeSpan ts);
        StartHandler postDelayTestStartHandler;

        Process python;
        private System.Windows.Forms.Timer timerScoreFollowing;  
        private int scoreFollowingState;
        private ZContext context;
        private ZSocket subscriber;
        private ZSocket publisher;
        private ZSocket responder_init;

        const int stopped   = 1;     /** Currently stopped */
        const int playing   = 2;     /** Currently playing music */
        const int paused    = 3;     /** Currently paused */
        const int initStop  = 4;     /** Transitioning from playing to stop */
        const int initPause = 5;     /** Transitioning from playing to pause */

        private ArrayList barIdxToDuration = new ArrayList();
        private List<NoteWithTime> notesTimesMap;

        private Stopwatch watch; // TODO: REMOVE        

        /// <summary>
        /// construct
        /// </summary>
        public SeeScoreSample()
        {
            InitializeComponent();
            SeeScore.Version version = SeeScore.SS.GetVersion();
            // ensure .lo has leading zero if < 10
            versionLabel.Text = "SeeScore v" + version.hi + "." + ((version.lo < 10) ? "0" + version.lo : "" +version.lo);
            bpmLabel.Text = "" + kDefaultTempoBPM;
            beatLabel.Hide();
            barControl.SetSSView(seeScoreView);

            /* Initialize the timer used for score following */
            timerScoreFollowing = new System.Windows.Forms.Timer();
            timerScoreFollowing.Enabled = false;
            // Need to run at a higher frquency than the backend (otherwise, 
            // we may not capture the instructions).
            timerScoreFollowing.Interval = 50;  
            timerScoreFollowing.Tick += new EventHandler(TimerCallbackScoreFollowing);
            scoreFollowingState = stopped; 
            
            // Create the ZMQ sockets, open the TCP ports.
            // Publisher only here
            context = new ZContext();          
            publisher = new ZSocket(context, ZSocketType.PUB);
            publisher.Bind("tcp://127.0.0.1:5556");       
                        
            responder_init = new ZSocket(context, ZSocketType.REP);
            responder_init.Bind("tcp://127.0.0.1:5557");

        }

        private async void ReadFileAsync(string filename) // asynchronous file read
        {
            bool isCompressed = filename.EndsWith(".mxl");
            byte[] result = null;
            if (isCompressed)
            {
                using (FileStream fileStream = File.Open(filename, FileMode.Open))
                {
                    using (System.IO.Compression.ZipArchive zip = new System.IO.Compression.ZipArchive(fileStream, ZipArchiveMode.Read))
                    {
                        foreach (ZipArchiveEntry entry in zip.Entries)
                        {
                            if (entry.Name.EndsWith(".xml") && entry.Name != "container.xml")
                            {
                                Stream stream = entry.Open();
                                using (MemoryStream memoryStream = new MemoryStream()) // copy to memorystream so we can find the length and allocate result
                                {
                                    await stream.CopyToAsync(memoryStream);
                                    result = memoryStream.ToArray();
                                }
                                break;
                            }
                        }
                    }
                }
            }
            else
            {
                using (FileStream sourceStream = File.Open(filename, FileMode.Open))
                {
                    result = new byte[sourceStream.Length];
                    await sourceStream.ReadAsync(result, 0, (int)sourceStream.Length);
                }
            }
            if (result != null)
            {
                loadedFile = filename;
                LoadData(result);
            }
            else
                loadedFile = null;
        }

        /// <summary>
        /// Callback class for PlayData passes back values for user-defined tempo / tempo scaling
        /// </summary>
        class UTempo : SeeScore.PlayData.IUserTempo
        {
            public UTempo(SeeScoreSample sss)
            { this.sss = sss; }

            /// <summary>
            /// used where absolute tempo is not defined in XML file
            /// </summary>
            /// <returns>absolute user-defined tempo in beats per minute</returns>
            public int GetUserTempo()
            {
                int rval = (int)(kDefaultTempoBPM * (float)sss.tempoSlider.Value / 100F);
                return rval;
            }

            /// <summary>
            /// used to scale tempo value defined in MusicXML file
            /// </summary>
            /// <returns>scaling - default is 1.0</returns>
            public float GetUserTempoScaling()
            {
                return (float)sss.tempoSlider.Value / 100F;
            }

            private SeeScoreSample sss;
        }

        /// <summary>
        /// handle events from SSView
        /// </summary>
        private class AppNotifierImpl : SeeScore.IAppNotifier
        {
            private SeeScoreSample app;
            public AppNotifierImpl(SeeScoreSample app)
            {
                this.app = app;
            }

            public void ReceivedClickInBar(int barIndex)
            {
                app.MovedCursor(barIndex);
            }

            public void Update()
            {
                app.barControl.UpdateCursor();
            }
        }

        private void LoadData(byte[] data)
        {
            try
            {       
                seeScoreView.ClearAwaitLayoutCompletion();// careful to await termination of layout thread if mid-layout
                if (score != null)
                {
                    score.Dispose();
                    score = null;
                }
                if (notifier != null)
                    notifier.Stop();
                axWindowsMediaPlayer1.Ctlcontrols.stop();
                axWindowsMediaPlayer1.URL = null;
                SeeScore.LoadOptions loadOptions = new SeeScore.LoadOptions(SeeScore.K.Key, false, true);
                score = SeeScore.SS.LoadXMLData(data, loadOptions);
                SeeScore.LoadWarning[] warnings = score.GetLoadWarnings();
                foreach (SeeScore.LoadWarning w in warnings)
                {
                    System.Console.WriteLine(" error " + w.warning + " in " + w.element + " in part:" + w.partIndex + " bar:" + w.barIndex);
                
                }
                tempoSlider.Value = 100;
                seeScoreView.SetScore(score, new AppNotifierImpl(this));
                midiPlayData_cache = null;
                this.zoomSlider.Value = 100;

                // Create MIDI file for playing
                midiFileName = loadedFile.Substring(0, openFileDialog1.FileName.Length - 4) + ".mid";                        
                GetMIDIFile();

                // Initialise the score follower
                InitialiseScoreFollower(midiFileName);
            }
            catch (SeeScore.LoadException ex)
            {
                Console.WriteLine("failed to load file: " + ex.Message);
            }
            catch (SeeScore.ScoreException ex)
            {
                Console.WriteLine("Failed to create midi file " + ex.Message);
            }
        }

        SeeScore.PlayData.IPlayData CreateMidiPlayData()
        {
            SeeScore.PlayData.IPlayData pd = SeeScore.PlayData.PD.Create(score, new UTempo(this));
            double rate = pd.HasDefinedTempo() ? pd.TempoAtStart().bpm : kDefaultTempoBPM; // use default 80 if no tempo defined in file
            bpmLabel.Text = "" + rate.ToString("F0");

            notifier = new SeeScore.Notifier(pd);
            if (kUsingNoteCursor)
                notifier.SetNoteHandler(new NoteHandler(this)); // note cursor

            notifier.SetBarChangeHandler(new BarEventHandler(this)); // bar cursor
            notifier.SetBeatHandler(new BeatHandler(this));
            notifier.SetEndHandler(new EndHandler(this));
            return pd;
        }

        SeeScore.PlayData.IPlayData GetMIDIPlayData()
        {
            if (midiPlayData_cache == null)
                midiPlayData_cache = CreateMidiPlayData();
            return midiPlayData_cache;
        }

        private void DisposeMIDI()
        {
            midiPlayData_cache = null;
        }

        private String GetMIDIFile()
        {
            if (midiPlayData_cache == null)
            {
                midiPlayData_cache = CreateMidiPlayData();
                midiPlayData_cache.CreateMidiFile(midiFileName);
            }
            return midiFileName;
        }

        void MoveNoteCursor(SeeScore.Notifier.PartNote[] notes)
        {
	        foreach (SeeScore.Notifier.PartNote note in notes) // normally this will not need to iterate over the whole chord, but will exit as soon as it has a valid xpos
	        {
		        if (note.note.start >= 0) // ignore cross-bar tied notes
		        {
			        float xpos = this.seeScoreView.NoteXPos(note.note);
			        if (xpos > 0) // noteXPos returns 0 if the note isn't found in the layout (it might be in a part which is not shown)
			        {
                        this.seeScoreView.ShowCursorAtXpos(xpos, note.note.startBarIndex, SeeScore.ScrollType.Bar);
				        return; // abandon iteration
			        }
		        }
	        }
	        Console.WriteLine("MoveNoteCursor failed");
        }

        private void openFileDialog1_FileOk(object sender, CancelEventArgs e)
        {
            loadedFile = ((OpenFileDialog)sender).FileName;
            ReadFileAsync(((OpenFileDialog)sender).FileName);
        }

        private void open_Click(object sender, EventArgs e)
        {
            openFileDialog1.ShowDialog();
        }

        private void zoomSlider_ValueChanged(object sender, EventArgs e)
        {
            float zoom = ((TrackBar)sender).Value / (float)100.0;
            zoomLabel.Text = "" + zoom;
            if (seeScoreView != null)
            {
                seeScoreView.ChangeZoom(zoom);
            }
        }

        private void transposeControl_ValueChanged(object sender, EventArgs e)
        {
            if (score == null)
                return;
            if (notifier != null)
                notifier.Stop();
            bool isPlaying = (axWindowsMediaPlayer1.playState == WMPPlayState.wmppsPlaying);
            if (isPlaying)// if playing stop
            {
                axWindowsMediaPlayer1.Ctlcontrols.stop();
            }
            axWindowsMediaPlayer1.URL = null;
            score.SetTranspose((int)((NumericUpDown)sender).Value);
            DisposeMIDI(); // need to regenerate transosed MIDI file
            seeScoreView.ShowCursorAtBar(0, SeeScore.CursorType.Line, SeeScore.ScrollType.Bar);
            seeScoreView.RequestRelayout();
            if (isPlaying)
                axWindowsMediaPlayer1.URL = GetMIDIFile(); // auto-start
        }

        private class BarEventHandler : SeeScore.Notifier.IEventHandler
        {
            private readonly SeeScoreSample sss;

            public BarEventHandler(SeeScoreSample sss)
            { this.sss = sss; }

            public void Event(int barIndex, bool countIn)
            {
                if (!kUsingNoteCursor)
                {
                    sss.seeScoreView.ShowCursorAtBar(barIndex, countIn ? SeeScore.CursorType.Line : SeeScore.CursorType.Rect, SeeScore.ScrollType.Bar);
                }
            }
        }

        private class PlayerStartDispatch : SeeScore.Dispatch
        {
            private delegate void SSSDelegate(SeeScoreSample sss);
            private static void StartPlayer(SeeScoreSample sss)
            {
                sss.axWindowsMediaPlayer1.Ctlcontrols.play();
            }
            public PlayerStartDispatch(DateTime when, SeeScoreSample sss)
            : base(when)
            {
                this.sss = sss;
            }
            protected override void Elapsed()
            {
                object[] args = new object[1];
                args[0] = sss;
                Dispatcher.CurrentDispatcher.Invoke(new SSSDelegate(StartPlayer), args); // call StartPlayer on main thread
            }
            private SeeScoreSample sss;
        }

        class DelayedStarter
        {
            public DelayedStarter(SeeScoreSample sss, int barIndex, bool countIn)
            {
                this.sss = sss;
                this.barIndex = barIndex;
                this.countIn = countIn;
            }

            public void postDelayTestStartHandler(TimeSpan startDelay)
            {
                sss.StartPlayerWithDelay(startDelay, barIndex, countIn);
            }

            private SeeScoreSample sss;
            private int barIndex;
            private bool countIn;
        }

        // called after testing mediaplayer start delay. We start the notifier and the mediaplayer with compensation for its start delay
        private void StartPlayerWithDelay(TimeSpan startDelay, int barIndex, bool countIn)
        {
            SeeScore.PlayData.IPlayData pd = GetMIDIPlayData();
            SeeScore.PlayData.IBarEnumerator barIter = pd.GetBarEnumerator();
            barIter.ChangeToBar(barIndex);
            SeeScore.PlayData.IBar bar = barIter.CurrentBar();
            DateTime mediaPlayerStartTime = DateTime.Now.AddMilliseconds(1000);
            DateTime notifierStartTime = mediaPlayerStartTime;
            if (countIn)
            {
                bar = bar.CreateCountIn();
                mediaPlayerStartTime = notifierStartTime.AddMilliseconds(bar.Duration() - startDelay.TotalMilliseconds);
            }
            else
            {
                mediaPlayerStartTime = notifierStartTime.AddMilliseconds(- startDelay.TotalMilliseconds);
            }
            int bar_ms = pd.durationUpToBar(barIndex);
            axWindowsMediaPlayer1.Ctlcontrols.currentPosition = (double)bar_ms / 1000.0F;
            new PlayerStartDispatch(mediaPlayerStartTime, this);
            notifier.StartAt(notifierStartTime, barIndex, countIn);// start count-in. MediaPlayer will be started after this
        }

        private void StartPlayAt(int barIndex, bool countIn)
        {
            axWindowsMediaPlayer1.URL = GetMIDIFile();
            axWindowsMediaPlayer1.Ctlcontrols.stop();
            SeeScore.PlayData.IPlayData pd = GetMIDIPlayData();
            int bar_ms = pd.durationUpToBar(barIndex);
            axWindowsMediaPlayer1.Ctlcontrols.currentPosition = (double)bar_ms / 1000.0F;
            DelayedStarter starter = new DelayedStarter(this, barIndex, countIn);
            testingDelay = true;
            mediaPlayerStartDelayTestStartTime = DateTime.Now;
            postDelayTestStartHandler = starter.postDelayTestStartHandler;
            axWindowsMediaPlayer1.Ctlcontrols.play();
        }

        private void playstop_Click(object sender, EventArgs e)
        {
            if (loadedFile != null)
            {
                if (axWindowsMediaPlayer1.playState == WMPPlayState.wmppsPlaying) // if playing stop
                {
                    axWindowsMediaPlayer1.Ctlcontrols.stop();
                }
                if (notifier != null && notifier.IsPlaying())
                {
                    notifier.Stop();
                }
                else // if not playing start
                {
                    int barIndex = seeScoreView.CursorBarIndex();
                    if (barIndex == score.NumBars() - 1) // if last bar reset to start
                    {
                        barIndex = 0;
                        axWindowsMediaPlayer1.Ctlcontrols.currentPosition = 0;
                    }
                    StartPlayAt(barIndex, true);
                }
            }
        }

        class BeatHandler : SeeScore.Notifier.IEventHandler
        {
            public BeatHandler(SeeScoreSample ss)
            {
                this.ss = ss;
            }
            public void Event(int index, bool countIn)
            {
                ss.beatLabel.Show();
                if (countIn)
                    ss.beatLabel.ForeColor = Color.Red;
                else
                    ss.beatLabel.ForeColor = Color.Black;
                ss.beatLabel.Text = "" + (index + 1);
            }
            private SeeScoreSample ss;
        }
        class EndHandler : SeeScore.Notifier.IEventHandler
        {
            public EndHandler(SeeScoreSample ss)
            {
                this.ss = ss;
            }
            public void Event(int index, bool countIn)
            {
                ss.beatLabel.Hide();
            }
            private SeeScoreSample ss;
        }
        class NoteHandler : SeeScore.Notifier.INoteHandler
        {
            public NoteHandler(SeeScoreSample ss)
            {
                this.ss = ss;
            }
	        public void startNotes(SeeScore.Notifier.PartNote[] notes)
	        {
		        ss.MoveNoteCursor(notes);
	        }
            public void endNote(SeeScore.Notifier.PartNote note)
	        {}
            private SeeScoreSample ss;
        };

        private void axWindowsMediaPlayer1_PlayStateChange(object sender, AxWMPLib._WMPOCXEvents_PlayStateChangeEvent e)
        {
            AxWMPLib.AxWindowsMediaPlayer mediaPlayer = (AxWMPLib.AxWindowsMediaPlayer)sender;
            WMPPlayState newState = mediaPlayer.playState;
            switch (newState)
            {
                case WMPPlayState.wmppsPlaying:
                    {
                        if (testingDelay)
                        {
                            /*
                             * The Windows Media Player has an unknown delay time from calling play to hearing the first note
                             * This code measures the delay from calling play() to the WMPPlayState.wmppsPlaying state change
                             * which is not necessarily the same, but it is better than doing nothing
                             * NB if you start after the first bar there is a delay for the music to start after the wmppsPlaying state change!
                             * This throws the notifier (ie moving cursor) sync out of the window!
                             * */
                            testingDelay = false;
                            TimeSpan mediaPlayerStartDelay = DateTime.Now.Subtract(mediaPlayerStartDelayTestStartTime);
                            mediaPlayer.Ctlcontrols.stop();
                            postDelayTestStartHandler(mediaPlayerStartDelay);
                            Console.WriteLine("Media Player start delay " + mediaPlayerStartDelay.TotalMilliseconds + "ms");
                        }
                        int barIndex = seeScoreView.CursorBarIndex();
                        playButton.Text = "Stop";
                    } break;

                case WMPPlayState.wmppsReady:
                case WMPPlayState.wmppsPaused:
                    break;

                case WMPPlayState.wmppsStopped:
                    {
                        playButton.Text = "Play";
                    }break;

                default:
                case WMPPlayState.wmppsUndefined:
                case WMPPlayState.wmppsTransitioning:
                    break;
            }
            lastPlayState = ((AxWMPLib.AxWindowsMediaPlayer)sender).playState;
        }

        private System.Windows.Forms.Timer _scrollingTimer = null;

        private void tempoSlider_Scroll(object sender, EventArgs e)
        {
            if (_scrollingTimer == null) // delay update of tempo until 500ms after finished moving slider
            {
                // Will tick every 500ms (change as required)
                _scrollingTimer = new System.Windows.Forms.Timer()
                {
                    Enabled = false,
                    Interval = 500,
                    Tag = (sender as TrackBar).Value
                };
                _scrollingTimer.Tick += (s, ea) =>
                {
                    // check to see if the value has changed since we last ticked
                    if ((sender as TrackBar).Value == (int)_scrollingTimer.Tag)
                    {
                        // scrolling has stopped so we are good to go ahead and do stuff
                        _scrollingTimer.Stop();

                        UpdateTempo((sender as TrackBar).Value / 100F);

                        _scrollingTimer.Dispose();
                        _scrollingTimer = null;
                    }
                    else
                    {
                        // record the last value seen
                        _scrollingTimer.Tag = (sender as TrackBar).Value;
                    }
                };
                _scrollingTimer.Start();
            }
        }

        void MovedCursor(int barIndex)
        {
            if (score != null)
            {
                if (axWindowsMediaPlayer1.playState == WMPPlayState.wmppsPlaying)
                    RestartAtBar(barIndex);
            }
        }

        void RestartAtBar(int barIndex)
        {
            axWindowsMediaPlayer1.Ctlcontrols.stop();
            notifier.Stop();
            StartPlayAt(barIndex, false);
        }

        int PositionToBarIndex(double position_s)
        {
            int ms = 0;
            int seqIndex = 0;
            int pos_ms = (int)(position_s * 1000);
            SeeScore.PlayData.IPlayData playData = GetMIDIPlayData();
            foreach (SeeScore.PlayData.IBar bar in playData)
            {
                ms += bar.Duration();
                if (ms >= pos_ms)
                    return seqIndex;
                ++seqIndex;
            }
            return seqIndex;
        }

        private void UpdateTempo(float tempoScaling)
        {
            if (score != null)
            {
                bool isPlaying = axWindowsMediaPlayer1.playState == WMPPlayState.wmppsPlaying;
                if (isPlaying)
                {
                    axWindowsMediaPlayer1.Ctlcontrols.stop();
                    notifier.Stop();
                }
                axWindowsMediaPlayer1.URL = null;
                SeeScore.PlayData.IPlayData playData = GetMIDIPlayData();
                double rate = (playData.HasDefinedTempo() ? playData.TempoAtStart().bpm : kDefaultTempoBPM) * tempoScaling;
                bpmLabel.Text = "" + rate.ToString("F0");
                playData.UpdateTempo();
                playData.ScaleMidiFileTempo(GetMIDIFile(), tempoScaling);
                if (isPlaying)
                {
                    int barIndex = seeScoreView.CursorBarIndex();
                    StartPlayAt(barIndex, false);
                }
            }
        }

        private void ignoreXMLLayout_CheckedChanged(object sender, EventArgs e)
        {
            seeScoreView.SetIgnoreXMLLayout(((CheckBox)sender).Checked);
        }

        private void SetNotesTimesMap()
        {
            GetMIDIPlayData();
            IBarEnumerator m_current_bar_iter = midiPlayData_cache.GetBarEnumerator(); 
            List<NoteWithTime> notes = new List<NoteWithTime>();
            int cumBarDuration = 0;
            while (!m_current_bar_iter.IsLast())
            {
                m_current_bar_iter.MoveNext();
                IBar bar = m_current_bar_iter.CurrentBar();
            
                if (midiPlayData_cache.NumParts() > 1)
                {
                    throw new System.InvalidOperationException("Should have one part only");
                }

                IPart part = bar.Part(0);
                foreach (Note note in part)
                {
                    Console.WriteLine(note.startBarIndex.ToString());
                    int noteTime = note.start + cumBarDuration;
                    notes.Add(new NoteWithTime(note, noteTime));                    
                }
                cumBarDuration = cumBarDuration + bar.Duration();
            }

            // sort in order of time
            notes.Sort(new NoteTimeComparer());  
            
            // Remove duplicate times (in case of equality, take the first note available)
            List<NoteWithTime> uniqueNotesTimes = new List<NoteWithTime>();
            uniqueNotesTimes.Add(notes[0]);
            for (int k = 1; k < notes.Count; k++)
            {
                //if (notes[k].noteTime != uniqueNotesTimes[uniqueNotesTimes.Count - 1].noteTime)
                //{
                //    uniqueNotesTimes.Add(notes[k]);
                //}
                uniqueNotesTimes.Add(notes[k]);
            }

            // TODO : remove negative times?
            notesTimesMap = uniqueNotesTimes;
        }

        internal class NoteWithTime
        {
            public NoteWithTime(Note n, int nTime)
            {
                note = n;
                noteTime = nTime;
            }

            public readonly Note note;
            public readonly int noteTime;
        }
        class NoteTimeComparer : IComparer<NoteWithTime>
        {
            public int Compare(NoteWithTime n1, NoteWithTime n2)
            {
                return n1.noteTime < n2.noteTime ? -1 : n1.noteTime == n2.noteTime ? 0 : +1;
            }
        }
        
        private Note FindClosestNote(int estPos)
        {
            if (notesTimesMap.Count == 0)
            {
                throw new System.ArgumentOutOfRangeException("The note <-> time map is empty");
            }

            // We assume that the map is properly sorted
            int idxClosestNote = -1;
            
            // Return the first note if we are before.
            if (estPos <= notesTimesMap[0].noteTime)
            {
                idxClosestNote = 0;
            }
            // Return the last note if we are after
            else if (estPos >= notesTimesMap[notesTimesMap.Count - 1].noteTime)            
            {
                idxClosestNote = notesTimesMap.Count - 1;
            }
            else
            {
                // Find the first note with a time greater than the estimated 
                // position and return the one just before.
                for (int k=1; k<notesTimesMap.Count; k++)
                {                    
                    if (estPos <= notesTimesMap[k].noteTime)
                    {
                        idxClosestNote = k - 1;
                        break;
                    }
                }                
            }

            if (idxClosestNote == -1)
            {
                throw new System.ArgumentNullException("Index not assigned, we should not have gotten here");
            }

            return(notesTimesMap[idxClosestNote].note);

        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            /* Dispose the Python subprocess*/
            if ((python != null) && (!python.HasExited)){
                python.Kill();
                python.Dispose();
            };
            
            /* Dispose the ZMQ sockets */
            subscriber.Dispose();
            publisher.Dispose();
            context.Dispose();
        }

        private void InitialiseScoreFollower(string filename)
        {        
            // Create the python process to run the python code and start it
            string filenamePythonScript = "C:/Users/Alexis/Source/TestPython2/AutomaticAudioTranscript/start_score_following.py";
            ProcessStartInfo info = new ProcessStartInfo();
            info.RedirectStandardError = true;
            info.UseShellExecute = false;
            info.FileName = "python3.exe";
            // We make sure the paths can handle spaces
            info.Arguments = "\"" + filenamePythonScript + "\"" + " " + "\"" + filename + "\"";
            python = new Process();
            python.StartInfo = info;        
            python.EnableRaisingEvents = true;
            python.Exited += new EventHandler(pythonProcess_Exited);
            python.Start();

            // Handle Exited event and display process information.
            void pythonProcess_Exited(object sender, System.EventArgs e)
            {   
                // Need to stop the score following timer, otherwise the window freezes when the python process is exited
                //player.timerScoreFollowing.Stop(); 
                MessageBox.Show(python.StandardError.ReadToEnd(), "Error in the python process",MessageBoxButtons.OK, MessageBoxIcon.Error);                                
            };

            // Block until we have received a request from the back-end 
            // (it takes a few secs to initialise and we don't want to do 
            // anything in the meantime).
            using (ZFrame request = responder_init.ReceiveFrame())
            {
                request.ReadString();
            }
            // REMOVE
            watch = Stopwatch.StartNew();
        } 

        /* The callback for the score follower. */            
        void TimerCallbackScoreFollowing(object sender, EventArgs args) {              
        
            // Check if we need to stop. We may stop for two reasons:
            // - the user has pressed stop.
            // - the backend has sent a stop instruction (e.g. end of track reached).
            if (scoreFollowingState == initStop) {
                // Stop the timer
                timerScoreFollowing.Stop();            
                scoreFollowingState = stopped;                

                // Send stop instruction to backend
                publisher.Send(new ZFrame("stop"));

                // Hide the cursor
                this.seeScoreView.HideCursor();
                return;
            }

            // Otherwise, we assume we can run normally
            publisher.Send(new ZFrame("start"));
            
            // Get the reply
            string reply;
            Int32 replyInt;            
            using (var replyFrame = subscriber.ReceiveFrame())
            {   
                reply = replyFrame.ReadString();
            }   
            Console.WriteLine(watch.ElapsedMilliseconds);
            
            // Try parsing the reply to see if it is a string or Int32
            bool isInt = Int32.TryParse(reply, out replyInt);
        
            // If Int32, then it is the current position
            if (isInt){                 
                // Based on the estimated position in ms, find the closest note
                Note closestNote = FindClosestNote(replyInt);

                float xpos = this.seeScoreView.NoteXPos(closestNote);
			    if (xpos > 0) // noteXPos returns 0 if the note isn't found in the layout (it might be in a part which is not shown)
			    {
                    this.seeScoreView.ShowCursorAtXpos(xpos, closestNote.startBarIndex, SeeScore.ScrollType.Bar);
				    return; // abandon iteration
			    }
                
            }
            // Otherwise it must be an instruction.
            else
            {
                if (reply == "stop")
                {
                    scoreFollowingState = initStop;
                }
                else{ 
                    throw new System.ArgumentException(string.Concat("Unexpected reply in the socket: ", reply));
                }
            }
        }
        
        private void ScoreFollowingButton_Click(object sender, EventArgs e)
        {            
            if (score == null) {
                return;
            }
            else if (scoreFollowingState == playing) {
                scoreFollowingState = initStop;
                scoreFollowingButton.Text = "Follow";
                return;
            }
            else if (scoreFollowingState == stopped) {

                SetNotesTimesMap(); // TODO : dispose afterward !!

                // Change status to play mode
                scoreFollowingButton.Text = "Stop";
                scoreFollowingState = playing;
                
                // Connect the subscriber
                // This is done here as it seems that the python publisher 
                // should have binded before the C# subscriber connects.                
                // In theory the connection order should not matter, but there
                // seems to be a ZMQ bug.               
                subscriber = new ZSocket(context, ZSocketType.SUB);  
                subscriber.Subscribe("");               
                subscriber.Conflate = true; // Keep only the last message
                subscriber.Connect("tcp://127.0.0.1:5555");  
            
                // Start the timer
                timerScoreFollowing.Start();
            }

        }


    }
}
