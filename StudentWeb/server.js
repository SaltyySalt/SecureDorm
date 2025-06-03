// Required dependencies
const express = require('express');
const mongoose = require('mongoose');
const bodyParser = require('body-parser');
const multer = require('multer');
const path = require('path');
const dotenv = require('dotenv');
const fs = require('fs');

// Load environment variables
dotenv.config();

// Import User model
const User = require('./models/user');

// Initialize app
const app = express();

// Set up middleware
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use(express.static('public'));
app.use(bodyParser.urlencoded({ extended: true }));

// Ensure uploads directory exists
const uploadDir = 'public/uploads';
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

// Configure multer for file upload
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, 'public/uploads/'),
  filename: (req, file, cb) => cb(null, Date.now() + '-' + file.originalname),
});
const upload = multer({ storage });

// Connect to MongoDB
mongoose.connect(process.env.MONGO_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true
})
  .then(() => console.log("✅ MongoDB connected"))
  .catch(err => console.error("❌ MongoDB connection error:", err));

// GET: registration page
app.get('/register', (req, res) => {
  const uid = req.query.uid;
  if (!uid) return res.status(400).send('❌ UID missing in query');
  console.log("🌐 UID from query:", uid);
  res.render('register', { uid });
});

// POST: form submission
app.post('/register', upload.single('photo'), async (req, res) => {
  const { uid, name, matric, phone } = req.body;
  console.log("📥 POST body:", req.body);
  console.log("📥 UID from body:", uid);

  if (!uid) return res.status(400).send('❌ UID is required');
  
  try {
    const { uid, name, matric, phone } = req.body;
    const photo = req.file ? `/uploads/${req.file.filename}` : null;

    console.log("📥 Received form data:", { uid, name, matric, phone, photo });

    if (!uid) {
      console.error("❗ UID missing from form");
      return res.status(400).send('UID is required');
    }

    const existing = await User.findOne({ uid });
    if (existing) {
      console.log("⚠️ Card already registered");
      return res.send('Card is already registered.');
    }

    const newUser = new User({ uid, name, matric, phone, photo });
    await newUser.save();

    console.log("✅ Registration successful!");
    res.send('✅ Registration successful!');
  } catch (error) {
    console.error("❌ Registration failed:", error);
    res.status(500).send('❌ Internal Server Error');
  }
});

// GET: all users
app.get('/users', async (req, res) => {
  try {
    const users = await User.find();
    res.json(users);
  } catch (err) {
    res.status(500).send('❌ Error fetching users');
  }
});

// Default route
app.get('/', (req, res) => {
  res.send('✅ Server is running');
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
