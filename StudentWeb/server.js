// // server.js

// // Import required modules
// require('dotenv').config();
// const express = require('express');
// const path = require('path');
// const mongoose = require('mongoose');
// const multer = require('multer');
// const dotenv = require('dotenv');

// // Security & Polishing Tips
// const helmet = require('helmet');
// app.use(helmet());

// // Load environment variables from .env file
// dotenv.config();

// // Initialize Express app
// const app = express();

// // Set the port
// const PORT = process.env.PORT || 3000;

// // Set up EJS as the templating engine
// app.set('view engine', 'ejs');
// app.set('views', path.join(__dirname, 'views'));

// // Middleware to parse URL-encoded bodies (as sent by HTML forms)
// app.use(express.urlencoded({ extended: true }));

// // Serve static files from the 'public' directory
// app.use(express.static(path.join(__dirname, 'public')));

// // Connect to MongoDB
// mongoose.connect(process.env.MONGODB_URI, {
//   useNewUrlParser: true,
//   useUnifiedTopology: true
// })
// .then(() => console.log('✅ Connected to MongoDB'))
// .catch(err => console.error('❌ MongoDB connection error:', err));

// // Define a Mongoose schema for registered users
// const userSchema = new mongoose.Schema({
//   uid: { type: String, unique: true, required: true },
//   name: { type: String, required: true },
//   photoPath: { type: String, required: true },
//   registeredAt: { type: Date, default: Date.now }
// });

// // Create a Mongoose model based on the schema
// const User = mongoose.model('User', userSchema);

// // Configure Multer for file uploads
// const storage = multer.diskStorage({
//   destination: function (req, file, cb) {
//     cb(null, 'public/uploads/'); // Save uploads in 'public/uploads/' directory
//   },
//   filename: function (req, file, cb) {
//     // Rename the file to include the UID and original extension
//     const ext = path.extname(file.originalname);
//     cb(null, `${req.body.uid}${ext}`);
//   }
// });

// const upload = multer({ storage: storage });

// // Route: Home page
// app.get('/', (req, res) => {
//   res.render('index'); // Renders 'views/index.ejs'
// });

// // Route: Display registration form
// app.post('/register', async (req, res) => {
//   const { uid, name, matricNo, phone } = req.body;
//   try {
//     const existingUser = await User.findOne({ uid });
//     if (existingUser) {
//       return res.send('This card has already been registered.');
//     }

//     const newUser = new User({ uid, name, matricNo, phone });
//     await newUser.save();

//     res.send(`✅ Registered successfully for ${name}`);
//   } catch (err) {
//     console.error(err);
//     res.status(500).send('Server error');
//   }
// });

// // Route: Handle registration form submission
// app.post('/register', upload.single('photo'), async (req, res) => {
//   const { uid, name } = req.body;
//   const photo = req.file;

//   if (!uid || !name || !photo) {
//     return res.status(400).send('All fields are required');
//   }

//   try {
//     // Check if the UID is already registered
//     const existingUser = await User.findOne({ uid });
//     if (existingUser) {
//       return res.send(`UID ${uid} is already registered to ${existingUser.name}`);
//     }

//     // Save the new user to the database
//     const newUser = new User({
//       uid,
//       name,
//       photoPath: `/uploads/${photo.filename}`
//     });

//     await newUser.save();

//     res.send(`Registration successful for ${name}`);
//   } catch (err) {
//     console.error('Error during registration:', err);
//     res.status(500).send('Internal Server Error');
//   }
// });

// // Start the server
// app.listen(PORT, () => {
//   console.log(`🚀 Server is running at http://localhost:${PORT}`);
// });


require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const bodyParser = require('body-parser');
const multer = require('multer');
const path = require('path');
const User = require('./models/User');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.static('public'));
app.use('/uploads', express.static('uploads'));
app.use(bodyParser.urlencoded({ extended: false }));
app.set('view engine', 'ejs');

// Connect to MongoDB
mongoose.connect(process.env.MONGODB_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true
})
.then(() => console.log('✅ Connected to MongoDB'))
.catch((err) => console.error('❌ MongoDB connection error:', err));

// Configure Multer for photo upload
const storage = multer.diskStorage({
  destination: './uploads/',
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  }
});
const upload = multer({ storage });

// Route to show registration form with UID
app.get('/register', (req, res) => {
  const uid = req.query.uid;
  if (!uid) return res.status(400).send('Missing UID in query.');
  res.render('register', { uid });
});

// Handle form submission
app.post('/register', upload.single('photo'), async (req, res) => {
  const { name, matric, phone, uid } = req.body;
  const photo = req.file ? req.file.path : null;

  try {
    const existing = await User.findOne({ uid });
    if (existing) return res.send('User already registered.');

    const newUser = new User({ uid, name, matric, phone, photo });
    await newUser.save();
    res.render('success', { name });
  } catch (error) {
    res.status(500).send('Registration failed.');
  }
});

// Home route
app.get('/', (req, res) => {
  res.send('Welcome to NFC Registration Portal');
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Server is running at http://localhost:${PORT}`);
});
