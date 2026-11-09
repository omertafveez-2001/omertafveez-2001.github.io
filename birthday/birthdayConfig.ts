// Personalize everything here, then run `npm run build` inside birthday/.
export const birthdayConfig = {
  nickname: 'Pretty Girl',
  greeting: 'Happy Birthday,',
  welcome: 'I made you a little something, because apparently annoying you wasn’t enough.',
  dedication: 'A tiny collection of things that make me think of you.',
  favoriteSound: {
    question: 'Do you know what my favorite sound is?',
    guesses: ['Music?', 'Rain?', 'My own voice?', 'Your laugh?'],
    answer: 'It’s your laugh. Obviously.',
    message: 'Especially when you’re trying to tell me to stop annoying you, but you can’t stop laughing.',
  },
  routine: {
    title: 'Let’s recreate your nighttime routine.',
    you: 'You: peacefully doing your nighttime routine.',
    me: 'Me: existing solely to annoy you.',
    responses: [
      'Okay okay, I’ll stop.',
      'Actually… one more thing.',
      'But you laugh so cutely 😭',
      'Fine, I’ll behave. For 5 seconds.',
      'You know I’m going to do it again, right?',
    ],
    ending: 'See? Making you laugh is literally my favorite hobby.',
  },
  quiz: {
    title: 'Your Very Important Urdu Education',
    subtitle: 'Because apparently I have appointed myself your Urdu teacher.',
    // Add more quiz words here if you want to expand the lesson later.
    words: [
      { word: 'Hatna', pronunciation: 'hut-naa', options: ['Pissed off', 'A very small potato', 'Please stop talking'], correct: 0, explanation: 'Pissed off — now you know.' },
    ],
    ending: '10/10 student. Mostly because your laugh makes every lesson worth it.',
  },
  letter: {
    invitation: 'One last thing, just for you.',
    salutation: 'To the girl with the cutest laugh,',
    poem: [
      'Happy birthday to the girl with the cutest laugh,\nThe one who makes me smile without even trying.',
      'From interrupting your nighttime routine,\nTo hearing you say ‘stoppp’ while laughing in between.',
      'From teaching you Urdu words you probably shouldn’t say,\nTo finding little reasons to make you laugh every day.',
      'And if there’s one thing I hope you always do,\nIt’s laugh that pretty laugh, and keep being you.',
    ],
    closing: 'Happy Birthday',
    signature: 'From,\nyour favorite annoyance',
  },
  colors: { blush: '#F8C8DC', cream: '#FFF9F5', rose: '#E88DA7', lavender: '#E9DDF5', ink: '#74394A' },
  decorations: { confetti: true },
};
