// models/User.js
const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
  uid: { type: String, required: true, unique: true },
  name: { type: String, required: true },
  matric: { type: String, required: true },
  phone: { type: String, required: true },
  photo: { type: String } // path to saved photo
});

module.exports = mongoose.model('User', userSchema);
