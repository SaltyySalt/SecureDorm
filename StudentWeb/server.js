// server.js

// ==============================
// 📦 Import Dependencies
// ==============================
const express = require('express');
const mongoose = require('mongoose');
const bodyParser = require('body-parser');
const multer = require('multer');
const path = require('path');
const dotenv = require('dotenv');
const fs = require('fs');
const User = require('./models/user'); // Mongoose model for users

// ==============================
// 📁 Load Environment Variables
// ==============================
dotenv.config(); // Loads variables from .env into process.env

// ==============================
// 🚀 Initialize App
// ==============================
const app = express();

// ==============================
// 🛠️ Middleware Setup
// ==============================

// Set EJS as the view engine
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Serve static files (e.g., images, CSS, JS) from /public
app.use(express.static('public'));

// Parse URL-encoded POST data
app.use(bodyParser.urlencoded({ extended: true }));

// ==============================
// 🔌 Connect to MongoDB
// ==============================
mongoose.set('bufferCommands', false); // Avoid buffering if DB is not ready
mongoose.connect(process.env.MONGO_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true
})
  .then(() => console.log("✅ MongoDB connected"))
  .catch(err => console.error("❌ MongoDB connection error:", err));

mongoose.connection.on('connected', () => {
  console.log('✅ Mongoose connected to DB');
});
mongoose.connection.on('error', err => {
  console.error('❌ Mongoose connection error:', err);
});
mongoose.connection.on('disconnected', () => {
  console.warn('⚠️ Mongoose disconnected');
});

// ==============================
// 📁 Ensure Upload Folder Exists
// ==============================
const uploadDir = 'public/uploads';
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

// ==============================
// 📤 Multer File Upload Setup
// ==============================
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'public/uploads/');
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + '-' + file.originalname);
  }
});
const upload = multer({ storage });

// ==============================
// 🧾 Routes
// ==============================

// 📄 Homepage
app.get('/', (req, res) => {
  res.send('Server is running');
});

app.get('/register', (req, res) => {
  const uid = req.query.uid;
  console.log("🌐 UID from query:", uid);
  res.render('register', { uid });
});

// 📋 Show Registration Form
app.get('/register', (req, res) => {
  const uid = req.query.uid; // Read UID from URL query (?uid=xxxx)
  console.log("🌐 UID from query:", uid);
  res.render('register', { uid }); // Pass UID to the EJS template
});

app.post('/register', upload.single('photo'), async (req, res) => {
  console.log("📥 POST body:", req.body);
  console.log("📥 UID from body:", req.body.uid);

// 📩 Handle Form Submission
app.post('/register', upload.single('photo'), async (req, res) => {
  const { uid, name, matric, phone } = req.body; // Form fields
  const photo = req.file ? `/uploads/${req.file.filename}` : null; // Uploaded file path

  console.log("📥 Form data received:", { uid, name, matric, phone, photo });
  
  try {
    if (!uid) return res.status(400).send('❌ UID is required');

    // Check if this UID is already registered
    const existing = await User.findOne({ uid });
    if (existing) return res.send('⚠️ Card is already registered.');

    // Save new user
    const newUser = new User({ uid, name, matric, phone, photo });
    await newUser.save();

    res.send('✅ Registration successful!');
  } catch (error) {
    console.error("❌ Registration failed:", error);
    res.status(500).send('❌ Internal Server Error');
  }
});

// 👀 Show All Registered Users (as JSON)
app.get('/users', async (req, res) => {
  try {
    const users = await User.find();
    res.json(users);
  } catch (err) {
    console.error('❌ Error fetching users:', err);
    res.status(500).send('❌ Error fetching users');
  }
});

// ==============================
// 🟢 Start Server
// ==============================
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
});
