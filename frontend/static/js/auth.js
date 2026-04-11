/* ── TrashDet — Authentication & Session (2-Factor with OTP) ─────────────── */

function authApp() {
  return {
    /* ── State ─────────────────────────────────────────────────────────── */
    user: null,
    token: localStorage.getItem('eco_token'),
    isLoggedIn: false,
    
    // Auth modals
    showAuthModal: false,
    authMode: 'login', // 'login' | 'register' | 'otp'
    
    // Forms
    loginData: { username: '', password: '' },
    registerData: { username: '', email: '', password: '' },
    authLoading: false,

    // OTP state
    otpDigits: ['', '', '', '', '', ''],
    otpEmailHint: '',
    otpUsername: '',
    otpCountdown: 0,
    otpResendAvailable: false,
    _otpInterval: null,

    // Password strength
    passwordRules: {
      minLength: false,
      hasUpper: false,
      hasLower: false,
      hasDigit: false,
      hasSpecial: false,
    },
    passwordScore: 0,
    showPasswordRules: false,

    /* ── Init ─────────────────────────────────────────────────────────── */
    async initAuth() {
      if (this.token) {
        await this.fetchMe();
      }
    },

    /* ── Password validation (real-time) ──────────────────────────────── */
    checkPasswordStrength(pw) {
      this.passwordRules.minLength = pw.length >= 8;
      this.passwordRules.hasUpper = /[A-Z]/.test(pw);
      this.passwordRules.hasLower = /[a-z]/.test(pw);
      this.passwordRules.hasDigit = /[0-9]/.test(pw);
      this.passwordRules.hasSpecial = /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?`~]/.test(pw);

      const passed = Object.values(this.passwordRules).filter(Boolean).length;
      this.passwordScore = passed; // 0-5
      this.showPasswordRules = pw.length > 0;
    },

    getPasswordStrengthLabel() {
      if (this.passwordScore <= 1) return { text: 'Foarte slabă', color: '#ef4444' };
      if (this.passwordScore <= 2) return { text: 'Slabă', color: '#f97316' };
      if (this.passwordScore <= 3) return { text: 'Medie', color: '#eab308' };
      if (this.passwordScore <= 4) return { text: 'Bună', color: '#22c55e' };
      return { text: 'Puternică', color: '#16a34a' };
    },

    /* ── Login Step 1: password → OTP ─────────────────────────────────── */
    async login() {
      this.authLoading = true;
      try {
        const formData = new URLSearchParams();
        formData.append('username', this.loginData.username);
        formData.append('password', this.loginData.password);

        const data = await fetchAPI('/api/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: formData
        });

        if (data.otp_required) {
          // Switch to OTP entry mode
          this.otpUsername = this.loginData.username;
          this.otpEmailHint = data.email_hint;
          this.otpDigits = ['', '', '', '', '', ''];
          this.authMode = 'otp';
          this.startOtpCountdown();
          showToast('Cod de verificare trimis pe email');
          // Focus first OTP input after DOM update
          this.$nextTick(() => {
            const first = document.querySelector('.otp-input-0');
            if (first) first.focus();
          });
        } else if (data.access_token) {
          // Direct token (fallback, shouldn't happen with OTP enabled)
          this.token = data.access_token;
          localStorage.setItem('eco_token', this.token);
          await this.fetchMe();
          this.showAuthModal = false;
          showToast(`Bine ai revenit, ${this.user.username}!`);
          window.dispatchEvent(new CustomEvent('eco:authChanged'));
        }
      } catch (e) {
        showToast(e.message, 'error');
      } finally {
        this.authLoading = false;
      }
    },

    /* ── Login Step 2: verify OTP ─────────────────────────────────────── */
    async verifyOTP() {
      const code = this.otpDigits.join('');
      if (code.length !== 6) {
        showToast('Introdu toate cele 6 cifre', 'error');
        return;
      }

      this.authLoading = true;
      try {
        const data = await fetchAPI('/api/auth/verify-otp', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username: this.otpUsername, code })
        });

        this.token = data.access_token;
        localStorage.setItem('eco_token', this.token);
        await this.fetchMe();
        this.showAuthModal = false;
        this.authMode = 'login';
        this.clearOtpState();
        showToast(`Bine ai revenit, ${this.user.username}!`);
        window.dispatchEvent(new CustomEvent('eco:authChanged'));
      } catch (e) {
        showToast(e.message, 'error');
        // Clear OTP inputs on error
        this.otpDigits = ['', '', '', '', '', ''];
        this.$nextTick(() => {
          const first = document.querySelector('.otp-input-0');
          if (first) first.focus();
        });
      } finally {
        this.authLoading = false;
      }
    },

    /* ── OTP digit input handling ─────────────────────────────────────── */
    handleOtpInput(index, event) {
      const value = event.target.value.replace(/\D/g, '');
      this.otpDigits[index] = value.slice(-1); // Only keep last digit
      event.target.value = this.otpDigits[index];

      // Auto-advance to next input
      if (value && index < 5) {
        const next = document.querySelector(`.otp-input-${index + 1}`);
        if (next) next.focus();
      }

      // Auto-submit when all 6 digits entered
      if (this.otpDigits.every(d => d !== '')) {
        this.verifyOTP();
      }
    },

    handleOtpKeydown(index, event) {
      if (event.key === 'Backspace' && !this.otpDigits[index] && index > 0) {
        const prev = document.querySelector(`.otp-input-${index - 1}`);
        if (prev) {
          this.otpDigits[index - 1] = '';
          prev.value = '';
          prev.focus();
        }
      }
    },

    handleOtpPaste(event) {
      event.preventDefault();
      const pasted = (event.clipboardData || window.clipboardData).getData('text').replace(/\D/g, '').slice(0, 6);
      for (let i = 0; i < 6; i++) {
        this.otpDigits[i] = pasted[i] || '';
        const input = document.querySelector(`.otp-input-${i}`);
        if (input) input.value = this.otpDigits[i];
      }
      // Focus last filled or first empty
      const nextEmpty = this.otpDigits.findIndex(d => d === '');
      const focusIdx = nextEmpty === -1 ? 5 : nextEmpty;
      const target = document.querySelector(`.otp-input-${focusIdx}`);
      if (target) target.focus();

      if (this.otpDigits.every(d => d !== '')) {
        this.verifyOTP();
      }
    },

    /* ── OTP countdown timer ──────────────────────────────────────────── */
    startOtpCountdown() {
      this.otpCountdown = 60;
      this.otpResendAvailable = false;
      if (this._otpInterval) clearInterval(this._otpInterval);
      this._otpInterval = setInterval(() => {
        this.otpCountdown--;
        if (this.otpCountdown <= 0) {
          clearInterval(this._otpInterval);
          this.otpResendAvailable = true;
        }
      }, 1000);
    },

    async resendOTP() {
      if (!this.otpResendAvailable) return;
      this.authLoading = true;
      try {
        const data = await fetchAPI('/api/auth/resend-otp', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username: this.otpUsername, code: '' })
        });
        this.otpEmailHint = data.email_hint;
        this.startOtpCountdown();
        this.otpDigits = ['', '', '', '', '', ''];
        showToast('Cod nou trimis pe email');
        this.$nextTick(() => {
          const first = document.querySelector('.otp-input-0');
          if (first) first.focus();
        });
      } catch (e) {
        showToast(e.message, 'error');
      } finally {
        this.authLoading = false;
      }
    },

    clearOtpState() {
      if (this._otpInterval) clearInterval(this._otpInterval);
      this.otpDigits = ['', '', '', '', '', ''];
      this.otpUsername = '';
      this.otpEmailHint = '';
      this.otpCountdown = 0;
      this.otpResendAvailable = false;
    },

    /* ── Register ─────────────────────────────────────────────────────── */
    async register() {
      // Client-side password check
      this.checkPasswordStrength(this.registerData.password);
      if (this.passwordScore < 5) {
        showToast('Parola nu îndeplinește toate cerințele', 'error');
        return;
      }

      this.authLoading = true;
      try {
        await fetchAPI('/api/auth/register', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(this.registerData)
        });

        showToast('Cont creat cu succes! Te poți loga acum.');
        this.authMode = 'login';
        this.loginData.username = this.registerData.username;
        this.showPasswordRules = false;
      } catch (e) {
        showToast(e.message, 'error');
      } finally {
        this.authLoading = false;
      }
    },

    /* ── Fetch current user ───────────────────────────────────────────── */
    async fetchMe() {
      try {
        this.user = await fetchAPI('/api/auth/me');
        this.isLoggedIn = true;
      } catch (e) {
        this.logout();
      }
    },

    /* ── Logout ───────────────────────────────────────────────────────── */
    logout() {
      this.token = null;
      this.user = null;
      this.isLoggedIn = false;
      localStorage.removeItem('eco_token');
      this.clearOtpState();
      showToast('Te-ai delogat cu succes.');
      window.dispatchEvent(new CustomEvent('eco:authChanged'));
    },

    openAuth(mode = 'login') {
      this.authMode = mode;
      this.showAuthModal = true;
      this.showPasswordRules = false;
    },

    backToLogin() {
      this.clearOtpState();
      this.authMode = 'login';
    }
  };
}
