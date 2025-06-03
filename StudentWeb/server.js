// server.js

require('dotenv').config(); // Load environment variables from .env

const express = require('express');
const mongoose = require('mongoose');
const multer = require('multer');
const bodyParser = require('body-parser');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000;

// MongoDB connection
mongoose.connect(process.env.MONGO_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true
}).then(() => {
  console.log("✅ MongoDB connected");
  // start server **after** connection
  app.listen(3000, () => {
    console.log("🚀 Server running at http://localhost:3000");
  });
}).catch((err) => {
  console.error("❌ MongoDB connection error:", err);
});


// Define user schema
const userSchema = new mongoose.Schema({
  uid: String,
  name: String,
  matricNo: String,
  phone: String,
  photoPath: String
});
const User = mongoose.model('User', userSchema);

// Middleware
app.set('view engine', 'ejs');
app.use(bodyParser.urlencoded({ extended: true }));
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// Setup multer for file upload
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => {
    const uniqueName = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, uniqueName + path.extname(file.originalname));
  }
});
const upload = multer({ storage: storage });

// Show registration form
app.get('/register', async (req, res) => {
  const { uid } = req.query;

  if (!uid) return res.status(400).send('UID not provided.');

  // Check if user already registered
  const user = await User.findOne({ uid });

  if (user) {
    res.send(`👋 Hello ${user.name}, you are already registered.`);
  } else {
    res.render('register', { uid });
  }
});

// Handle registration form submission
app.post('/register', upload.single('photo'), async (req, res) => {
  try {
    const { uid, name, matricNo, phone } = req.body;
    const photoPath = req.file ? req.file.path : null;

    if (!uid || !name || !matricNo || !phone || !photoPath) {
      return res.status(400).send('Missing required fields.');
    }

    // Check if already registered
    const existingUser = await User.findOne({ uid });
    if (existingUser) {
      return res.send(`⚠️ UID already registered to ${existingUser.name}`);
    }

    const user = new User({ uid, name, matricNo, phone, photoPath });
    await user.save();

    res.send(`✅ Registration successful for ${name}`);
  } catch (err) {
    console.error('❌ Registration error:', err);
    res.status(500).send('Internal Server Error');
  }
});

// Home redirect (optional)
app.get('/', (req, res) => {
  res.send('NFC Registration Portal is live');
});

app.listen(PORT, () => {
  console.log(`🚀 Server is running at http://localhost:${PORT}`);
});
