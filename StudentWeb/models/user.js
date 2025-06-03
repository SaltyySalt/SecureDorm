const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
  uid: { type: String, unique: true, required: true },
  name: String,
  matric: String,
  phone: String
});

module.exports = mongoose.model('User', userSchema);
