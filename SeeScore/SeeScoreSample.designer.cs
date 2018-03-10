namespace SeeScoreWin
{
    partial class SeeScoreSample
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(SeeScoreSample));
            this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
            this.openButton = new System.Windows.Forms.Button();
            this.zoomSlider = new System.Windows.Forms.TrackBar();
            this.ignoreXMLLayoutCheck = new System.Windows.Forms.CheckBox();
            this.versionLabel = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.zoomLabel = new System.Windows.Forms.Label();
            this.transposeControl = new System.Windows.Forms.NumericUpDown();
            this.label1 = new System.Windows.Forms.Label();
            this.panel1 = new System.Windows.Forms.Panel();
            this.seeScoreView = new SeeScore.SSView();
            this.panel2 = new System.Windows.Forms.Panel();
            this.scoreFollowingButton = new System.Windows.Forms.Button();
            this.beatLabel = new System.Windows.Forms.Label();
            this.label5 = new System.Windows.Forms.Label();
            this.bpmLabel = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.tempoSlider = new System.Windows.Forms.TrackBar();
            this.axWindowsMediaPlayer1 = new AxWMPLib.AxWindowsMediaPlayer();
            this.playButton = new System.Windows.Forms.Button();
            this.panel3 = new System.Windows.Forms.Panel();
            this.barControl = new SeeScore.BarControl();
            ((System.ComponentModel.ISupportInitialize)(this.zoomSlider)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.transposeControl)).BeginInit();
            this.panel1.SuspendLayout();
            this.panel2.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.tempoSlider)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.axWindowsMediaPlayer1)).BeginInit();
            this.panel3.SuspendLayout();
            this.SuspendLayout();
            // 
            // openFileDialog1
            // 
            this.openFileDialog1.FileOk += new System.ComponentModel.CancelEventHandler(this.openFileDialog1_FileOk);
            // 
            // openButton
            // 
            this.openButton.Location = new System.Drawing.Point(9, 7);
            this.openButton.Margin = new System.Windows.Forms.Padding(1);
            this.openButton.Name = "openButton";
            this.openButton.Size = new System.Drawing.Size(68, 33);
            this.openButton.TabIndex = 0;
            this.openButton.Text = "open";
            this.openButton.UseVisualStyleBackColor = true;
            this.openButton.Click += new System.EventHandler(this.open_Click);
            // 
            // zoomSlider
            // 
            this.zoomSlider.Location = new System.Drawing.Point(295, 2);
            this.zoomSlider.Margin = new System.Windows.Forms.Padding(4);
            this.zoomSlider.Maximum = 300;
            this.zoomSlider.Minimum = 30;
            this.zoomSlider.Name = "zoomSlider";
            this.zoomSlider.Size = new System.Drawing.Size(411, 56);
            this.zoomSlider.TabIndex = 2;
            this.zoomSlider.Value = 100;
            this.zoomSlider.ValueChanged += new System.EventHandler(this.zoomSlider_ValueChanged);
            // 
            // ignoreXMLLayoutCheck
            // 
            this.ignoreXMLLayoutCheck.AutoSize = true;
            this.ignoreXMLLayoutCheck.Location = new System.Drawing.Point(20, 15);
            this.ignoreXMLLayoutCheck.Margin = new System.Windows.Forms.Padding(4);
            this.ignoreXMLLayoutCheck.Name = "ignoreXMLLayoutCheck";
            this.ignoreXMLLayoutCheck.Size = new System.Drawing.Size(149, 21);
            this.ignoreXMLLayoutCheck.TabIndex = 3;
            this.ignoreXMLLayoutCheck.Text = "Ignore XML Layout";
            this.ignoreXMLLayoutCheck.UseVisualStyleBackColor = true;
            this.ignoreXMLLayoutCheck.CheckedChanged += new System.EventHandler(this.ignoreXMLLayout_CheckedChanged);
            // 
            // versionLabel
            // 
            this.versionLabel.AutoSize = true;
            this.versionLabel.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.versionLabel.Location = new System.Drawing.Point(16, 64);
            this.versionLabel.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.versionLabel.Name = "versionLabel";
            this.versionLabel.Size = new System.Drawing.Size(168, 25);
            this.versionLabel.TabIndex = 4;
            this.versionLabel.Text = "SeeScore V??.??";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label2.Location = new System.Drawing.Point(228, 7);
            this.label2.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(68, 25);
            this.label2.TabIndex = 5;
            this.label2.Text = "Zoom:";
            // 
            // zoomLabel
            // 
            this.zoomLabel.AutoSize = true;
            this.zoomLabel.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.zoomLabel.Location = new System.Drawing.Point(720, 7);
            this.zoomLabel.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.zoomLabel.Name = "zoomLabel";
            this.zoomLabel.Size = new System.Drawing.Size(39, 25);
            this.zoomLabel.TabIndex = 6;
            this.zoomLabel.Text = "1.0";
            // 
            // transposeControl
            // 
            this.transposeControl.Location = new System.Drawing.Point(1036, 14);
            this.transposeControl.Margin = new System.Windows.Forms.Padding(4);
            this.transposeControl.Maximum = new decimal(new int[] {
            12,
            0,
            0,
            0});
            this.transposeControl.Minimum = new decimal(new int[] {
            12,
            0,
            0,
            -2147483648});
            this.transposeControl.Name = "transposeControl";
            this.transposeControl.Size = new System.Drawing.Size(53, 22);
            this.transposeControl.TabIndex = 7;
            this.transposeControl.ValueChanged += new System.EventHandler(this.transposeControl_ValueChanged);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(948, 16);
            this.label1.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(80, 17);
            this.label1.TabIndex = 8;
            this.label1.Text = "Transpose:";
            // 
            // panel1
            // 
            this.panel1.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel1.AutoScroll = true;
            this.panel1.Controls.Add(this.seeScoreView);
            this.panel1.Location = new System.Drawing.Point(-7, 68);
            this.panel1.Margin = new System.Windows.Forms.Padding(4);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(1124, 610);
            this.panel1.TabIndex = 9;
            // 
            // seeScoreView
            // 
            this.seeScoreView.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.seeScoreView.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(255)))), ((int)(((byte)(243)))));
            this.seeScoreView.Location = new System.Drawing.Point(4, 4);
            this.seeScoreView.Margin = new System.Windows.Forms.Padding(4);
            this.seeScoreView.Name = "seeScoreView";
            this.seeScoreView.Size = new System.Drawing.Size(1108, 607);
            this.seeScoreView.TabIndex = 1;
            // 
            // panel2
            // 
            this.panel2.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel2.Controls.Add(this.scoreFollowingButton);
            this.panel2.Controls.Add(this.beatLabel);
            this.panel2.Controls.Add(this.label5);
            this.panel2.Controls.Add(this.bpmLabel);
            this.panel2.Controls.Add(this.label3);
            this.panel2.Controls.Add(this.tempoSlider);
            this.panel2.Controls.Add(this.axWindowsMediaPlayer1);
            this.panel2.Controls.Add(this.playButton);
            this.panel2.Controls.Add(this.zoomSlider);
            this.panel2.Controls.Add(this.versionLabel);
            this.panel2.Controls.Add(this.ignoreXMLLayoutCheck);
            this.panel2.Controls.Add(this.label2);
            this.panel2.Controls.Add(this.zoomLabel);
            this.panel2.Location = new System.Drawing.Point(13, 683);
            this.panel2.Margin = new System.Windows.Forms.Padding(4);
            this.panel2.Name = "panel2";
            this.panel2.Size = new System.Drawing.Size(1084, 110);
            this.panel2.TabIndex = 10;
            // 
            // scoreFollowingButton
            // 
            this.scoreFollowingButton.Location = new System.Drawing.Point(951, 64);
            this.scoreFollowingButton.Margin = new System.Windows.Forms.Padding(4);
            this.scoreFollowingButton.Name = "scoreFollowingButton";
            this.scoreFollowingButton.Size = new System.Drawing.Size(100, 28);
            this.scoreFollowingButton.TabIndex = 12;
            this.scoreFollowingButton.Text = "Follow";
            this.scoreFollowingButton.UseVisualStyleBackColor = true;
            this.scoreFollowingButton.Click += new System.EventHandler(this.ScoreFollowingButton_Click);
            // 
            // beatLabel
            // 
            this.beatLabel.AutoSize = true;
            this.beatLabel.Font = new System.Drawing.Font("Microsoft Sans Serif", 27.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.beatLabel.ForeColor = System.Drawing.Color.Red;
            this.beatLabel.Location = new System.Drawing.Point(819, 28);
            this.beatLabel.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.beatLabel.Name = "beatLabel";
            this.beatLabel.Size = new System.Drawing.Size(77, 54);
            this.beatLabel.TabIndex = 13;
            this.beatLabel.Text = "99";
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Font = new System.Drawing.Font("Georgia", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label5.Location = new System.Drawing.Point(763, 64);
            this.label5.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(39, 17);
            this.label5.TabIndex = 12;
            this.label5.Text = "BPM";
            // 
            // bpmLabel
            // 
            this.bpmLabel.AutoSize = true;
            this.bpmLabel.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.bpmLabel.Location = new System.Drawing.Point(720, 59);
            this.bpmLabel.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.bpmLabel.Name = "bpmLabel";
            this.bpmLabel.Size = new System.Drawing.Size(45, 25);
            this.bpmLabel.TabIndex = 11;
            this.bpmLabel.Text = "999";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label3.Location = new System.Drawing.Point(217, 58);
            this.label3.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(80, 25);
            this.label3.TabIndex = 10;
            this.label3.Text = "Tempo:";
            // 
            // tempoSlider
            // 
            this.tempoSlider.Location = new System.Drawing.Point(295, 54);
            this.tempoSlider.Margin = new System.Windows.Forms.Padding(4);
            this.tempoSlider.Maximum = 400;
            this.tempoSlider.Minimum = 25;
            this.tempoSlider.Name = "tempoSlider";
            this.tempoSlider.Size = new System.Drawing.Size(411, 56);
            this.tempoSlider.TabIndex = 9;
            this.tempoSlider.Value = 100;
            this.tempoSlider.Scroll += new System.EventHandler(this.tempoSlider_Scroll);
            // 
            // axWindowsMediaPlayer1
            // 
            this.axWindowsMediaPlayer1.Enabled = true;
            this.axWindowsMediaPlayer1.Location = new System.Drawing.Point(700, 0);
            this.axWindowsMediaPlayer1.Margin = new System.Windows.Forms.Padding(4);
            this.axWindowsMediaPlayer1.Name = "axWindowsMediaPlayer1";
            this.axWindowsMediaPlayer1.OcxState = ((System.Windows.Forms.AxHost.State)(resources.GetObject("axWindowsMediaPlayer1.OcxState")));
            this.axWindowsMediaPlayer1.Size = new System.Drawing.Size(135, 29);
            this.axWindowsMediaPlayer1.TabIndex = 8;
            this.axWindowsMediaPlayer1.Visible = false;
            this.axWindowsMediaPlayer1.PlayStateChange += new AxWMPLib._WMPOCXEvents_PlayStateChangeEventHandler(this.axWindowsMediaPlayer1_PlayStateChange);
            // 
            // playButton
            // 
            this.playButton.Location = new System.Drawing.Point(951, 15);
            this.playButton.Margin = new System.Windows.Forms.Padding(4);
            this.playButton.Name = "playButton";
            this.playButton.Size = new System.Drawing.Size(100, 28);
            this.playButton.TabIndex = 7;
            this.playButton.Text = "Play";
            this.playButton.UseVisualStyleBackColor = true;
            this.playButton.Click += new System.EventHandler(this.playstop_Click);
            // 
            // panel3
            // 
            this.panel3.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel3.Controls.Add(this.barControl);
            this.panel3.Controls.Add(this.openButton);
            this.panel3.Controls.Add(this.label1);
            this.panel3.Controls.Add(this.transposeControl);
            this.panel3.Location = new System.Drawing.Point(13, 2);
            this.panel3.Margin = new System.Windows.Forms.Padding(4);
            this.panel3.Name = "panel3";
            this.panel3.Size = new System.Drawing.Size(1096, 62);
            this.panel3.TabIndex = 11;
            // 
            // barControl
            // 
            this.barControl.Location = new System.Drawing.Point(112, 6);
            this.barControl.Margin = new System.Windows.Forms.Padding(4);
            this.barControl.Name = "barControl";
            this.barControl.Size = new System.Drawing.Size(808, 52);
            this.barControl.TabIndex = 9;
            // 
            // SeeScoreSample
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1113, 798);
            this.Controls.Add(this.panel3);
            this.Controls.Add(this.panel2);
            this.Controls.Add(this.panel1);
            this.Margin = new System.Windows.Forms.Padding(1);
            this.Name = "SeeScoreSample";
            this.Text = "SeeScore Sample App";
            ((System.ComponentModel.ISupportInitialize)(this.zoomSlider)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.transposeControl)).EndInit();
            this.panel1.ResumeLayout(false);
            this.panel2.ResumeLayout(false);
            this.panel2.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.tempoSlider)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.axWindowsMediaPlayer1)).EndInit();
            this.panel3.ResumeLayout(false);
            this.panel3.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.OpenFileDialog openFileDialog1;
        private System.Windows.Forms.Button openButton;
        private SeeScore.SSView seeScoreView;
        private System.Windows.Forms.TrackBar zoomSlider;
        private System.Windows.Forms.CheckBox ignoreXMLLayoutCheck;
        private System.Windows.Forms.Label versionLabel;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label zoomLabel;
        private System.Windows.Forms.NumericUpDown transposeControl;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.Panel panel2;
        private System.Windows.Forms.Panel panel3;
        private System.Windows.Forms.Button playButton;
        private AxWMPLib.AxWindowsMediaPlayer axWindowsMediaPlayer1;
        private System.Windows.Forms.Label beatLabel;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label bpmLabel;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TrackBar tempoSlider;
        private SeeScore.BarControl barControl;
        private System.Windows.Forms.Button scoreFollowingButton;
    }
}

