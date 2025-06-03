require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const bodyParser = require('body-parser');
const path = require('path');
const User = require('./models/User');

const app = express();

app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static('public'));
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

mongoose.connect(process.env.MONGO_URI)
  .then(() => console.log('✅ MongoDB Connected'))
  .catch(err => console.error('❌ MongoDB Error:', err));

// Show form
app.get('/register', async (req, res) => {
  const uid = req.query.uid;
  const user = await User.findOne({ uid });
  if (user) {
    res.send(`You have already registered as ${user.name}. Welcome!`);
  } else {
    res.render('register', { uid });
  }
});

// Handle form submission
app.post('/register', async (req, res) => {
  const { uid, name, matric, phone } = req.body;
  try {
    await User.create({ uid, name, matric, phone });
    res.send('✅ Registration successful!');
  } catch (error) {
    res.send('❌ Registration failed. UID might already be registered.');
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
});
