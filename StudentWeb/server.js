const express = require('express');
const mongoose = require('mongoose');
const bodyParser = require('body-parser');
const multer = require('multer');
const path = require('path');
const dotenv = require('dotenv');
const User = require('./models/user');

dotenv.config();
const app = express();

app.set('view engine', 'ejs');
app.use(express.static('public'));
app.use(bodyParser.urlencoded({ extended: true }));

mongoose.connect(process.env.MONGO_URI)
  .then(() => console.log("✅ MongoDB connected"))
  .catch(err => console.error("❌ MongoDB connection error:", err));


const fs = require('fs');
const uploadDir = 'public/uploads';

if (!fs.existsSync(uploadDir)){
    fs.mkdirSync(uploadDir, { recursive: true });
}

// File upload setup
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'public/uploads/');
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + '-' + file.originalname);
  }
});
const upload = multer({ storage });

// View engine setup
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// GET registration form
app.get('/register', (req, res) => {
  const uid = req.query.uid;
  res.render('register', { uid });
});

// POST registration form
app.post('/register', upload.single('photo'), async (req, res) => {
  const { uid, name, matric, phone } = req.body;
  const photo = req.file ? `/uploads/${req.file.filename}` : null;

  console.log("📥 Received data:", { uid, name, matric, phone, photo });

  try {
    if (!uid) return res.status(400).send('UID is required');
    
    const existing = await User.findOne({ uid });
    if (existing) return res.send('Card is already registered.');

    const newUser = new User({ uid, name, matric, phone, photo });
    await newUser.save();

    res.send('✅ Registration successful!');
  } catch (error) {
    console.error("❌ Registration failed:", error);
    res.status(500).send('❌ Internal Server Error');
  }
});


app.get('/users', async (req, res) => {
  try {
    const users = await User.find();
    res.json(users); // Shows all users as JSON
  } catch (err) {
    res.status(500).send('❌ Error fetching users');
  }
});

app.get('/', (req, res) => {
  res.send('Server is running');
});

// Server start
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
