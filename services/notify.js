const nodemailer = require('nodemailer');

const from = process.env.MAIL_FROM;
const to = process.env.MAIL_TO;
const host = process.env.SMTP_HOST;
const port = Number(process.env.SMTP_PORT || 587);
const user = process.env.SMTP_USER;
const pass = process.env.SMTP_PASS;

let transporter = null;
if (host && user && pass && from && to) {
  transporter = nodemailer.createTransport({
    host, port, secure: port === 465, auth: { user, pass }
  });
}

async function sendViolationMail(payload) {
  if (!transporter) return;
  const subject = `[PPE Alert] Missing: ${payload.missingItems.join(', ')}`;
  const text = `Client: ${payload.clientId || '-'}\nMissing: ${payload.missingItems.join(', ')}\nAt: ${new Date().toISOString()}`;
  try { await transporter.sendMail({ from, to, subject, text }); }
  catch (e) { console.warn('sendViolationMail failed', e.message); }
}

module.exports = { sendViolationMail };
