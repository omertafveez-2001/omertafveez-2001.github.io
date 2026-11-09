'use client';

import { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion, MotionConfig, useReducedMotion } from 'framer-motion';
import { ArrowLeft, ArrowRight, Check, Mail, Music2, RotateCcw, Sparkles, Volume2, CloudRain, Mic, Smile } from 'lucide-react';
import { birthdayConfig as config } from '../birthdayConfig';
import { Cake, Vanity } from '../components/Illustrations';

const chapters = ['A little hello', 'My favorite sound', 'A familiar scene', 'Urdu, obviously', 'Just for you'];
const guessIcons = [Music2, CloudRain, Mic, Smile];

function NextButton({ children, onClick }: { children: React.ReactNode; onClick: () => void }) {
  return <motion.button className="primary" onClick={onClick} whileHover={{ y: -2 }} whileTap={{ scale: .97 }}>{children}<ArrowRight size={17} aria-hidden="true" /></motion.button>;
}
function Eyebrow({ children }: { children: React.ReactNode }) {
  return <p className="eyebrow"><span />{children}<span /></p>;
}

export default function BirthdayPage() {
  const [step, setStep] = useState(0);
  const [unlocked, setUnlocked] = useState(0);
  const [guess, setGuess] = useState<number | null>(null);
  const [laughs, setLaughs] = useState(0);
  const [question, setQuestion] = useState(0);
  const [answers, setAnswers] = useState<(number | null)[]>(config.quiz.words.map(() => null));
  const [quizDone, setQuizDone] = useState(false);
  const [letterOpen, setLetterOpen] = useState(false);
  const reducedMotion = useReducedMotion();
  const heading = useRef<HTMLHeadingElement>(null);
  const mounted = useRef(false);

  useEffect(() => {
    if (!mounted.current) { mounted.current = true; return; }
    const timer = setTimeout(() => {
      heading.current?.focus({ preventScroll: true });
      window.scrollTo({ top: 0, behavior: reducedMotion ? 'instant' : 'smooth' });
    }, reducedMotion ? 0 : 220);
    return () => clearTimeout(timer);
  }, [step, reducedMotion]);

  function go(next: number) { setUnlocked(value => Math.max(value, next)); setStep(next); }
  function replay() {
    setGuess(null); setLaughs(0); setQuestion(0); setAnswers(config.quiz.words.map(() => null));
    setQuizDone(false); setLetterOpen(false); setUnlocked(0); setStep(0);
  }
  const word = config.quiz.words[question];
  const selected = answers[question];
  const answered = selected !== null;

  return <MotionConfig reducedMotion="user"><div className="birthday" style={{ '--blush': config.colors.blush, '--cream': config.colors.cream, '--rose': config.colors.rose, '--lavender': config.colors.lavender, '--ink': config.colors.ink } as React.CSSProperties}>
    <a className="skip-link" href="#surprise">Skip to your surprise</a>
    <div className="paper-grain" aria-hidden="true" />
    <header className="site-header">
      <a className="wordmark" href="#surprise" aria-label="A little something, birthday surprise"><span>a little something</span></a>
      <span className="header-note">made just for you </span>
    </header>
    <main id="surprise">
      <nav className="chapter-nav" aria-label="Your surprise progress">
        <span className="chapter-count">0{step + 1} <span>/ 05</span></span>
        <div className="progress-dots">{chapters.map((chapter, index) => <button key={chapter} disabled={index > unlocked} aria-label={`Chapter ${index + 1}: ${chapter}${index > unlocked ? ' (not opened yet)' : ''}`} aria-current={step === index ? 'step' : undefined} className={`progress-dot ${index <= unlocked ? 'unlocked' : ''} ${step === index ? 'active' : ''}`} onClick={() => go(index)}><span /></button>)}</div>
        <span className="chapter-label">{chapters[step]}</span>
      </nav>
      <AnimatePresence mode="wait" initial={false}>
        <motion.section key={step} className={`story chapter-${step}`} aria-labelledby="chapter-title" initial={{ opacity: 0, y: reducedMotion ? 0 : 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: reducedMotion ? 0 : -10 }} transition={{ duration: reducedMotion ? 0 : .3 }}>
          {step === 0 && <>
            <Eyebrow>A birthday-sized little note</Eyebrow>
            <div className="welcome-art">
              <div className="art-halo" />
              <span className="hand-note note-left">make a wish <span>↘</span></span>
              <Cake />
              <span className="hand-note note-right">this one’s<br />all yours</span>
              {config.decorations.confetti && <div className="confetti" aria-hidden="true">{Array.from({ length: 12 }, (_, i) => <motion.i key={i} style={{ left: `${12 + (i * 19) % 78}%`, background: ['#E8BBCB', '#D8CAE5', '#DFC397'][i % 3], width: i % 2 ? 4 : 6 }} initial={{ y: -30, opacity: 0, rotate: 0 }} animate={{ y: [0, 75 + i * 9], opacity: [0, .7, 0], rotate: i * 32 }} transition={{ duration: reducedMotion ? 0 : 3, delay: i * .09 }} />)}</div>}
            </div>
            <h1 id="chapter-title" ref={heading} tabIndex={-1}>{config.greeting}<br /><em>{config.nickname}</em></h1>
            <p className="intro">{config.welcome}</p>
            <NextButton onClick={() => go(1)}>Open your little surprise <Mail size={17} aria-hidden="true" /></NextButton>
            <p className="tiny-note">a few little moments. a whole lot of you.</p>
          </>}
          {step === 1 && <>
            <Eyebrow>01 / The little things</Eyebrow>
            <div className="sound-doodle" aria-hidden="true"><div className="sound-wave">{[12, 24, 38, 24, 51, 33, 17, 42, 26, 13].map((height, i) => <span key={i} style={{ height, animationDelay: `${i * .12}s` }}/>)}</div></div>
            <h1 id="chapter-title" ref={heading} tabIndex={-1}>{config.favoriteSound.question}</h1>
            <p className="intro">Go on, take a guess. I’ll wait.</p>
            <div className="guess-grid">{config.favoriteSound.guesses.map((label, i) => { const Icon = guessIcons[i] || Volume2; return <motion.button whileTap={{ scale: .97 }} className={`guess ${guess === i ? 'chosen' : ''}`} aria-pressed={guess === i} key={label} onClick={() => setGuess(i)}><Icon size={20} strokeWidth={1.5} aria-hidden="true"/>{label}<span>{guess === i ? '✓' : '↗'}</span></motion.button>; })}</div>
            <div aria-live="polite" className="reveal-space">{guess !== null && <motion.div className="answer-note" initial={{ opacity: 0, y: reducedMotion ? 0 : 8 }} animate={{ opacity: 1, y: 0 }}><span className="hand-note">{guess === 3 ? 'you knew, didn’t you?' : 'a good guess, but…'}</span><h2 className="type-reveal">{config.favoriteSound.answer}</h2><p>{config.favoriteSound.message}</p></motion.div>}</div>
            {guess !== null && <NextButton onClick={() => go(2)}>Speaking of annoying you…</NextButton>}
          </>}
          {step === 2 && <>
            <Eyebrow>02 / Sound familiar?</Eyebrow>
            <h1 id="chapter-title" ref={heading} tabIndex={-1}>{config.routine.title}</h1>
            <div className="vanity-scene"><Vanity laughs={laughs} /><span className="hand-note vanity-note">a very peaceful<br />evening. allegedly.</span></div>
            <div className="scene-lines"><p>{config.routine.you}</p><p>{config.routine.me}</p></div>
            <motion.button className="stop-button" whileTap={{ scale: .92, rotate: -2 }} whileHover={{ rotate: 2 }} onClick={() => setLaughs(value => value + 1)}>STOPPPP <span aria-hidden="true">😭</span></motion.button>
            <div className="laugh-response" aria-live="polite"><AnimatePresence mode="wait"><motion.p key={laughs} initial={{ opacity: 0, y: reducedMotion ? 0 : 5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: .15 }}>{laughs ? config.routine.responses[(laughs - 1) % config.routine.responses.length] : 'Go on. You know what to do.'}</motion.p></AnimatePresence></div>
            <span className="laugh-count"><Smile size={15} aria-hidden="true"/> little laughs collected <b>{String(laughs).padStart(2, '0')}</b></span>
            {laughs >= 5 && <motion.div className="game-ending" initial={{ opacity: 0 }} animate={{ opacity: 1 }}><p>{config.routine.ending}</p><NextButton onClick={() => go(3)}>Time for a little Urdu lesson</NextButton></motion.div>}
            {laughs < 5 && <p className="tiny-note">{5 - laughs} more {5 - laughs === 1 ? 'tap' : 'taps'} until I (probably) behave.</p>}
          </>}
          {step === 3 && <>
            <Eyebrow>03 / Class is in session</Eyebrow>
            <h1 id="chapter-title" ref={heading} tabIndex={-1}>{config.quiz.title}</h1>
            <p className="intro">{config.quiz.subtitle}</p>
            {!quizDone ? <div className="quiz-card">
              <div className="quiz-topline"><span>THE VERY UNOFFICIAL URDU CLUB</span><span>{question + 1} / {config.quiz.words.length}</span></div>
              <span className="hand-note">today’s very important word</span><h2 className="urdu-word">{word.word}</h2><p className="pronunciation">pronounced “{word.pronunciation}”</p>
              <div className="quiz-options">{word.options.map((option, i) => <motion.button key={`${question}-${i}`} disabled={answered} whileTap={{ scale: .98 }} aria-pressed={selected === i} className={`quiz-option ${answered && word.correct === i ? 'correct' : ''} ${answered && selected === i && i !== word.correct ? 'incorrect' : ''}`} onClick={() => setAnswers(values => values.map((value, index) => index === question ? i : value))}><span className="option-letter">{String.fromCharCode(65 + i)}</span>{option}{answered && word.correct === i && <Check size={17} aria-label="Correct answer"/>}</motion.button>)}</div>
              <div aria-live="polite">{answered && <motion.div className="quiz-feedback" initial={{ opacity: 0, scale: .96 }} animate={{ opacity: 1, scale: 1 }}><p><b>{selected === word.correct ? 'Look at you! A natural. ✧' : 'Cute answer. Here’s the real one:'}</b><br />{word.explanation}</p><button className="text-button" onClick={() => { if (question + 1 < config.quiz.words.length) setQuestion(value => value + 1); else setQuizDone(true); }}>{question + 1 < config.quiz.words.length ? 'Next word' : 'Hand in my homework'}<ArrowRight size={16} aria-hidden="true"/></button></motion.div>}</div>
            </div> : <motion.div className="quiz-finished" initial={{ opacity: 0, y: reducedMotion ? 0 : 10 }} animate={{ opacity: 1, y: 0 }}><div className="grade">10<span>/10</span><Sparkles aria-hidden="true"/></div><span className="hand-note">teacher’s favorite, obviously.</span><h2>{config.quiz.ending}</h2><p className="tiny-note">{answers.filter((answer, index) => answer === config.quiz.words[index].correct).length} of {config.quiz.words.length} correct. Full marks for being you.</p><NextButton onClick={() => go(4)}>There’s one last thing</NextButton></motion.div>}
          </>}
          {step === 4 && <>
            <Eyebrow>04 / A little birthday note</Eyebrow>
            <h1 id="chapter-title" ref={heading} tabIndex={-1}>{config.letter.invitation}</h1>
            <p className="intro">Some things deserve a little more than a birthday text.</p>
            <div className={`envelope-scene ${letterOpen ? 'is-open' : ''}`}>
              <motion.button className="envelope" onClick={() => setLetterOpen(value => !value)} aria-expanded={letterOpen} aria-controls="birthday-letter" aria-label={letterOpen ? 'Close your birthday letter' : 'Open your birthday letter'} whileHover={{ y: -4 }} whileTap={{ scale: .98 }}>
                <span className="envelope-paper" /><span className="envelope-flap"/><span className="envelope-front"/><span className="envelope-seal"><Sparkles size={20} strokeWidth={1.5} aria-hidden="true"/></span><span className="envelope-address">for {config.nickname.toLowerCase()},<br /><i>just for you.</i></span>
              </motion.button><p className="hand-note envelope-hint">{letterOpen ? 'a little note to keep forever' : 'a tiny tap to open ↗'}</p>
            </div>
            <AnimatePresence>{letterOpen && <motion.article id="birthday-letter" className="letter" initial={{ opacity: 0, height: reducedMotion ? 'auto' : 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: reducedMotion ? 'auto' : 0 }} transition={{ duration: reducedMotion ? 0 : .6 }}><div className="letter-inner"><p className="salutation">{config.letter.salutation}</p>{config.letter.poem.map(stanza => <p className="stanza" key={stanza}>{stanza}</p>)}<div className="letter-divider" aria-hidden="true" /><h2>{config.letter.closing}</h2><p className="signature">{config.letter.signature}</p></div></motion.article>}</AnimatePresence>
            {letterOpen && <motion.div className="ending" initial={{ opacity: 0 }} animate={{ opacity: 1 }}><p>Here’s to more laughs, more little moments, more you.</p><button className="text-button" onClick={replay}><RotateCcw size={14} aria-hidden="true"/>A little smile, all over again</button></motion.div>}
          </>}
        </motion.section>
      </AnimatePresence>
      {step > 0 && <button className="back-button" onClick={() => go(step - 1)}><ArrowLeft size={14} aria-hidden="true"/> a little step back</button>}
    </main>
  </div></MotionConfig>;
}
