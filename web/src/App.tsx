import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Shield,
  Mail,
  MessageSquare,
  Globe,
  AlertTriangle,
  Lock,
  Link,
  Eye,
  Search,
  CheckCircle,
  XCircle,
  Fingerprint,
  ArrowRight,
  ShieldCheck,
  Zap
} from 'lucide-react';

// ==================== TYPES ====================

interface ScanReason {
  text: string;
  weight?: number;
}

interface ScanResult {
  verdict: string;
  score: number;
  stage: number;
  reasons?: ScanReason[];
  latency_ms?: number;
  note?: string | null;
  error?: boolean;
  message?: string;
}

interface RecentScan {
  url: string;
  verdict: string;
}

interface SessionStats {
  scans: number;
  phishing: number;
  totalLatency: number;
}

// ==================== CONFIGURATION ====================

const MOCK_RESPONSE: ScanResult = {
  verdict: 'phishing',
  score: 0.73,
  stage: 2,
  reasons: [
    {
      text: 'URL contains suspicious login keyword',
      weight: 1
    }
  ],
  latency_ms: 120
};

const IMAGES = {
  hero: 'https://images.unsplash.com/photo-1550751827-4bd374c3f58b?w=1200&q=85',
  phishing: 'https://images.unsplash.com/photo-1563013544-824ae1b704d3?w=900&q=85',
  email: 'https://images.unsplash.com/photo-1596526131083-e8c633c948d2?w=700&q=80',
  security: 'https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=900&q=85'
};

const phishingTypes = [
  {
    icon: Mail,
    title: 'Email Phishing',
    description:
      'Fraudulent emails impersonate trusted brands, banks and organizations to steal passwords and financial information.',
    image: IMAGES.email
  },
  {
    icon: MessageSquare,
    title: 'Smishing',
    description:
      'Attackers send deceptive SMS messages containing malicious links or urgent requests for sensitive information.',
    image: IMAGES.phishing
  },
  {
    icon: Globe,
    title: 'Typosquatting',
    description:
      'Fake websites use slightly misspelled domain names to imitate legitimate websites and deceive visitors.',
    image: IMAGES.hero
  },
  {
    icon: Shield,
    title: 'Spear Phishing',
    description:
      'Targeted phishing attacks designed around specific individuals or organizations using personalized messages.',
    image: IMAGES.security
  }
];

const protectionTips = [
  {
    icon: Link,
    title: 'Check URLs Carefully',
    description:
      'Look for spelling mistakes, unusual domains and suspicious subdomains before opening links.'
  },
  {
    icon: Lock,
    title: 'Protect Your Passwords',
    description:
      'Never share passwords or OTPs. Use strong and unique passwords for your accounts.'
  },
  {
    icon: Fingerprint,
    title: 'Enable Two-Factor Authentication',
    description:
      'Add an extra layer of security to your important online accounts.'
  },
  {
    icon: Mail,
    title: 'Verify Suspicious Messages',
    description:
      'Contact organizations through their official websites instead of replying to suspicious messages.'
  },
  {
    icon: Eye,
    title: 'Watch Out for Urgency',
    description:
      'Be careful with messages threatening account suspension or promising unrealistic rewards.'
  },
  {
    icon: ShieldCheck,
    title: 'Scan Suspicious Links',
    description:
      'Use PhishGuard to analyze suspicious URLs before interacting with them.'
  }
];

const attackSteps = [
  {
    number: '01',
    title: 'Impersonation',
    description:
      'Attackers pretend to be trusted banks, companies, services or individuals.'
  },
  {
    number: '02',
    title: 'Manipulation',
    description:
      'They create urgency, fear or attractive offers to encourage quick action.'
  },
  {
    number: '03',
    title: 'Data Theft',
    description:
      'Victims may unknowingly submit passwords, OTPs or personal information.'
  },
  {
    number: '04',
    title: 'Exploitation',
    description:
      'Stolen information may be used for fraud, identity theft or account takeover.'
  }
];

// ==================== MAIN APP ====================

export default function App() {
  const [showAnnounce, setShowAnnounce] = useState(true);
  const [url, setUrl] = useState('');
  const [valid, setValid] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ScanResult | null>(null);

  const [session, setSession] = useState<SessionStats>({
    scans: 0,
    phishing: 0,
    totalLatency: 0
  });

  const [recent, setRecent] = useState<RecentScan[]>([]);
  const [backendDown, setBackendDown] = useState(false);

  const mockMode = import.meta.env.VITE_MOCK === 'true';

  // ==================== BACKEND HEALTH ====================

  useEffect(() => {
    fetch('/api/health')
      .then((response) => {
        if (!response.ok) {
          throw new Error('Backend unavailable');
        }
        return response.json();
      })
      .then((data) => {
        setBackendDown(!data.models_loaded);
      })
      .catch(() => {
        setBackendDown(true);
      });
  }, []);

  // ==================== URL VALIDATION ====================

  const validateUrl = (value: string): boolean => {
    try {
      const test = /^https?:\/\//i.test(value)
        ? value
        : `https://${value}`;

      const parsed = new URL(test);

      return (
        (parsed.protocol === 'https:' ||
          parsed.protocol === 'http:') &&
        parsed.hostname.includes('.') &&
        !parsed.hostname.startsWith('.')
      );
    } catch {
      return false;
    }
  };

  // ==================== SCAN FUNCTION ====================

  const handleScan = async () => {
    const trimmedUrl = url.trim();

    if (!trimmedUrl || !validateUrl(trimmedUrl)) {
      setValid(false);
      return;
    }

    setValid(true);
    setLoading(true);
    setResult(null);

    const payloadUrl = /^https?:\/\//i.test(trimmedUrl)
      ? trimmedUrl
      : `https://${trimmedUrl}`;

    try {
      let data: ScanResult;

      if (mockMode) {
        data = await new Promise<ScanResult>((resolve) => {
          setTimeout(() => resolve(MOCK_RESPONSE), 800);
        });
      } else {
        const response = await fetch('/api/check', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            url: payloadUrl
          })
        });

        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }

        data = await response.json();
      }

      if (!data || !data.verdict) {
        throw new Error('Invalid response from server.');
      }

      setResult(data);
      setBackendDown(false);

      setSession((previous) => ({
        scans: previous.scans + 1,
        phishing:
          previous.phishing +
          (data.verdict.toLowerCase() === 'phishing' ? 1 : 0),
        totalLatency:
          previous.totalLatency + (Number(data.latency_ms) || 0)
      }));

      setRecent((previous) =>
        [
          {
            url: payloadUrl,
            verdict: data.verdict
          },
          ...previous
        ].slice(0, 5)
      );

    } catch (error) {
      console.error('Scanning failed:', error);
      setBackendDown(true);

      setResult({
        verdict: 'error',
        score: 0,
        stage: 0,
        error: true,
        message:
          error instanceof Error
            ? error.message
            : 'Unable to scan this URL.'
      });
    } finally {
      setLoading(false);
    }
  };

  const formatNumber = (value: number) =>
    new Intl.NumberFormat().format(value);

  const avgLatency = session.scans
    ? Math.round(session.totalLatency / session.scans)
    : 0;

  const scrollToScanner = () => {
    document.getElementById('scan')?.scrollIntoView({
      behavior: 'smooth'
    });
  };

  // ==================== RENDER ====================

  return (
    <div className="min-h-screen bg-white text-slate-900 font-sans">

      {/* ANNOUNCEMENT BAR */}

      <AnimatePresence>
        {showAnnounce && (
          <motion.div
            initial={{ y: -50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: -50, opacity: 0 }}
            className="bg-blue-700 text-white px-4 py-3 flex justify-between items-center gap-3"
          >
            <span className="text-sm">
              PhishGuard Academic Project — Automated results are estimates.
            </span>

            <button
              onClick={() => setShowAnnounce(false)}
              aria-label="Close announcement"
            >
              ✕
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* NAVIGATION */}

      <nav className="sticky top-0 z-40 bg-white/95 backdrop-blur border-b border-slate-200 px-5 md:px-10 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">

          <a href="#" className="flex items-center gap-2">
            <div className="bg-blue-700 text-white p-2 rounded-lg">
              <ShieldCheck size={25} />
            </div>

            <span className="text-2xl font-extrabold text-slate-900">
              Phish<span className="text-blue-700">Guard</span>
            </span>
          </a>

          <div className="hidden md:flex items-center gap-7 text-sm font-semibold text-slate-700">
            <a href="#about" className="hover:text-blue-700">About</a>
            <a href="#statistics" className="hover:text-blue-700">Threats</a>
            <a href="#types" className="hover:text-blue-700">Attack Types</a>
            <a href="#protection" className="hover:text-blue-700">Safety Tips</a>
          </div>

          <button
            onClick={scrollToScanner}
            className="bg-blue-700 text-white px-5 py-2.5 rounded-lg font-semibold hover:bg-blue-800 transition"
          >
            Scan a URL →
          </button>

        </div>
      </nav>

      {/* HERO SECTION */}

      <section className="bg-white relative overflow-hidden">
        <div className="max-w-7xl mx-auto px-5 md:px-10 py-16 md:py-24 grid md:grid-cols-2 gap-12 items-center">

          <div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
            >
              <span className="inline-flex items-center gap-2 bg-blue-50 text-blue-700 px-4 py-2 rounded-full text-sm font-semibold">
                <ShieldCheck size={17} />
                Smart Phishing Detection
              </span>

              <h1 className="text-4xl md:text-6xl font-extrabold leading-tight mt-6 text-slate-900">
                Protect Yourself From
                <span className="text-blue-700"> Online Threats</span>
              </h1>

              <p className="text-slate-600 text-lg mt-6 leading-relaxed">
                Detect suspicious websites before they deceive you.
                Analyze URLs, understand potential threats and browse
                with greater confidence using PhishGuard.
              </p>

              <div className="flex flex-wrap gap-4 mt-8">
                <button
                  onClick={scrollToScanner}
                  className="bg-blue-700 text-white px-7 py-3 rounded-xl font-bold flex items-center gap-2 hover:bg-blue-800 transition"
                >
                  Check a URL <ArrowRight size={18} />
                </button>

                <a
                  href="#about"
                  className="border border-slate-300 text-slate-800 px-7 py-3 rounded-xl font-semibold hover:border-blue-700 hover:text-blue-700 transition"
                >
                  Learn About Phishing
                </a>
              </div>

              <div className="flex flex-wrap gap-6 mt-9 text-sm text-slate-600">
                <span className="flex items-center gap-2">
                  <CheckCircle className="text-green-600" size={17} />
                  Easy to use
                </span>

                <span className="flex items-center gap-2">
                  <CheckCircle className="text-green-600" size={17} />
                  Automated analysis
                </span>

                <span className="flex items-center gap-2">
                  <CheckCircle className="text-green-600" size={17} />
                  Security awareness
                </span>
              </div>
            </motion.div>
          </div>

          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.7 }}
            className="relative"
          >
            <img
              src={IMAGES.hero}
              alt="Cybersecurity and digital protection"
              className="w-full h-80 md:h-[440px] object-cover rounded-3xl shadow-xl"
            />

            <div className="absolute -bottom-5 -left-3 md:-left-7 bg-white border border-slate-200 shadow-xl p-4 rounded-xl flex items-center gap-3">
              <div className="bg-green-50 text-green-600 p-3 rounded-full">
                <ShieldCheck size={28} />
              </div>

              <div>
                <p className="font-bold text-slate-900">Stay Protected</p>
                <p className="text-sm text-slate-500">
                  Think before you click
                </p>
              </div>
            </div>
          </motion.div>

        </div>
      </section>

      {/* WHAT IS PHISHING */}

      <section id="about" className="px-5 md:px-10 py-20 bg-slate-50">
        <div className="max-w-7xl mx-auto">

          <div className="grid md:grid-cols-2 gap-12 items-center">

            <div>
              <span className="text-blue-700 font-bold text-sm uppercase tracking-widest">
                Know the Threat
              </span>

              <h2 className="text-3xl md:text-4xl font-extrabold mt-3 text-slate-900">
                What is <span className="text-blue-700">Phishing?</span>
              </h2>

              <p className="text-slate-600 leading-relaxed mt-6">
                Phishing is a type of cyberattack where criminals
                impersonate trusted organizations, websites or
                individuals to trick people into revealing sensitive
                information.
              </p>

              <p className="text-slate-600 leading-relaxed mt-4">
                Attackers commonly use fake websites, emails, SMS
                messages and social media messages to steal login
                credentials, banking details, OTPs and personal data.
                Some attacks may also deliver malicious software.
              </p>

              <div className="grid grid-cols-2 gap-4 mt-8">

                <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
                  <Lock className="text-blue-700 mb-2" size={25} />
                  <h3 className="font-bold text-slate-900">Credential Theft</h3>
                  <p className="text-sm text-slate-600 mt-1">
                    Stealing usernames and passwords.
                  </p>
                </div>

                <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
                  <AlertTriangle className="text-blue-700 mb-2" size={25} />
                  <h3 className="font-bold text-slate-900">Online Fraud</h3>
                  <p className="text-sm text-slate-600 mt-1">
                    Financial and identity-related scams.
                  </p>
                </div>

              </div>
            </div>

            <div>
              <img
                src={IMAGES.phishing}
                alt="Digital security and online fraud awareness"
                className="w-full h-80 md:h-[420px] object-cover rounded-2xl shadow-lg"
                loading="lazy"
              />

              <p className="text-xs text-slate-500 mt-3 text-center">
                Always verify a website before sharing sensitive information.
              </p>
            </div>

          </div>

          {/* HOW PHISHING WORKS */}

          <div className="mt-20">
            <div className="text-center mb-10">

              <span className="text-blue-700 font-bold text-sm uppercase tracking-widest">
                Attack Process
              </span>

              <h3 className="text-2xl md:text-3xl font-bold mt-3 text-slate-900">
                How Does a Phishing Attack Work?
              </h3>

              <p className="text-slate-600 mt-3">
                Understand the typical stages of a phishing attack.
              </p>

            </div>

            <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-5">

              {[...[
                {
                  number: '01',
                  title: 'Impersonation',
                  description: 'Attackers pretend to be trusted banks, companies, services or individuals.'
                },
                {
                  number: '02',
                  title: 'Manipulation',
                  description: 'They create urgency, fear or attractive offers to encourage quick action.'
                },
                {
                  number: '03',
                  title: 'Data Theft',
                  description: 'Victims may unknowingly submit passwords, OTPs or personal information.'
                },
                {
                  number: '04',
                  title: 'Exploitation',
                  description: 'Stolen information may be used for fraud, identity theft or account takeover.'
                }
              ].map((step, index) => (
                <motion.div
                  key={step.number}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm hover:shadow-lg transition"
                >
                  <span className="text-4xl font-extrabold text-blue-200">
                    {step.number}
                  </span>

                  <h4 className="text-xl font-bold mt-4 text-slate-900">
                    {step.title}
                  </h4>

                  <p className="text-slate-600 text-sm mt-3 leading-relaxed">
                    {step.description}
                  </p>
                </motion.div>
              ))]}
            </div>
          </div>

        </div>
      </section>

      {/* CYBER THREAT STATISTICS */}

      <section id="statistics" className="px-5 md:px-10 py-20 bg-white">
        <div className="max-w-7xl mx-auto">

          <div className="mb-12">
            <span className="text-blue-700 font-bold text-sm uppercase tracking-widest">
              Cyber Threat Intelligence
            </span>

            <h2 className="text-3xl md:text-4xl font-extrabold mt-3 text-slate-900">
              The Growing <span className="text-blue-700">Cyber Threat</span>
            </h2>

            <p className="text-slate-600 mt-4 max-w-3xl">
              Historical figures shown in your supplied reference image
              illustrate different aspects of deceptive websites and
              email threats.
            </p>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">

            {[{
                icon: AlertTriangle,
                value: '35.6%',
                title: 'Deceptive Links',
                description: 'Reported as a major method used by cyber actors in the reference report.'
              },
              {
                icon: Mail,
                value: '89%',
                title: 'Email Authentication',
                description: 'The reference reports unwanted messages passing SPF, DKIM or DMARC checks.'
              },
              {
                icon: Globe,
                value: '30%',
                title: 'New Domains',
                description: 'Share of detected threats featuring newly registered domains in the reference.'
              },
              {
                icon: Shield,
                value: '4.2M',
                title: 'Fraudulent Websites',
                description: 'Websites recorded by Bolster in 2022, according to your reference.'
              }].map((stat, index) => {
                const Icon = stat.icon;
                return (
                  <motion.div
                    key={stat.title}
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: index * 0.1 }}
                    className="bg-slate-50 p-7 rounded-2xl shadow-sm hover:shadow-xl border border-slate-200 transition"
                  >
                    <div className="bg-blue-100 text-blue-700 p-3 rounded-xl inline-flex">
                      <Icon size={28} />
                    </div>

                    <h3 className="text-3xl font-extrabold text-slate-900 mt-6">
                      {stat.value}
                    </h3>

                    <h4 className="font-bold text-lg mt-2 text-slate-900">
                      {stat.title}
                    </h4>

                    <p className="text-slate-600 text-sm mt-3 leading-relaxed">
                      {stat.description}
                    </p>
                  </motion.div>
                );
              })}
          </div>

          <p className="text-xs text-slate-500 text-right mt-5">
            Historical figures transcribed from your supplied reference image.
            Original reports and methodology should be verified before publication.
          </p>

        </div>
      </section>

      {/* TYPES OF PHISHING */}

      <section id="types" className="px-5 md:px-10 py-20 bg-slate-50">
        <div className="max-w-7xl mx-auto">

          <div className="text-center mb-12">

            <span className="text-blue-700 font-bold text-sm uppercase tracking-widest">
              Know the Techniques
            </span>

            <h2 className="text-3xl md:text-4xl font-extrabold mt-3 text-slate-900">
              Common Types of <span className="text-blue-700">Phishing</span>
            </h2>

            <p className="text-slate-600 mt-4 max-w-2xl mx-auto">
              Cybercriminals use different methods to deceive users.
              Learn to recognize these common attacks.
            </p>

          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">

            {phishingTypes.map((type, index) => {
              const Icon = type.icon;
              return (
                <motion.div
                  key={type.title}
                  initial={{ opacity: 0, y: 25 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-sm hover:shadow-xl transition"
                >
                  <img
                    src={type.image}
                    alt={type.title}
                    loading="lazy"
                    className="w-full h-40 object-cover"
                  />

                  <div className="p-6">
                    <div className="bg-blue-100 text-blue-700 inline-flex p-3 rounded-xl">
                      <Icon size={24} />
                    </div>

                    <h3 className="font-bold text-xl mt-4 text-slate-900">
                      {type.title}
                    </h3>

                    <p className="text-slate-600 text-sm mt-3 leading-relaxed">
                      {type.description}
                    </p>
                  </div>
                </motion.div>
              );
            })}

          </div>
        </div>
      </section>

      {/* SAFETY TIPS */}

      <section id="protection" className="px-5 md:px-10 py-20 bg-white">
        <div className="max-w-7xl mx-auto">

          <div className="grid md:grid-cols-2 gap-12 items-center">

            <div>
              <span className="text-blue-700 font-bold text-sm uppercase tracking-widest">
                Your Digital Safety
              </span>

              <h2 className="text-3xl md:text-4xl font-extrabold mt-3 text-slate-900">
                Stay Safe From <span className="text-blue-700">Phishing</span>
              </h2>

              <p className="text-slate-600 mt-4 leading-relaxed">
                A few simple security habits can help reduce your
                risk of falling victim to phishing attacks.
              </p>

              <div className="grid sm:grid-cols-2 gap-5 mt-9">

                {protectionTips.map((tip) => {
                  const Icon = tip.icon;
                  return (
                    <div key={tip.title} className="flex items-start gap-3">
                      <div className="bg-blue-50 text-blue-700 p-2 rounded-lg shrink-0">
                        <Icon size={21} />
                      </div>

                      <div>
                        <h3 className="font-bold text-slate-900">
                          {tip.title}
                        </h3>

                        <p className="text-sm text-slate-600 mt-1 leading-relaxed">
                          {tip.description}
                        </p>
                      </div>
                    </div>
                  );
                })}

              </div>
            </div>

            <div className="relative">
              <img
                src={IMAGES.security}
                alt="Cybersecurity protection"
                loading="lazy"
                className="w-full h-[430px] object-cover rounded-3xl shadow-xl"
              />

              <div className="absolute bottom-5 left-5 right-5 bg-white/95 text-slate-900 p-5 rounded-xl shadow-lg">
                <div className="flex items-center gap-3">
                  <ShieldCheck className="text-green-600" size={30} />

                  <div>
                    <h3 className="font-bold">
                      Think Before You Click
                    </h3>

                    <p className="text-sm text-slate-600">
                      Verify first. Stay protected.
                    </p>
                  </div>
                </div>
              </div>
            </div>

          </div>
        </div>
      </section>

      {/* URL SCANNER */}

      <section id="scan" className="px-5 md:px-10 py-20 bg-slate-50">
        <div className="max-w-4xl mx-auto">

          <div className="text-center mb-10">

            <div className="inline-flex bg-blue-100 text-blue-700 p-4 rounded-2xl">
              <Search size={32} />
            </div>

            <h2 className="text-3xl md:text-4xl font-extrabold mt-5 text-slate-900">
              Check a <span className="text-blue-700">Suspicious URL</span>
            </h2>

            <p className="text-slate-600 mt-3">
              Enter a website address to analyze it using PhishGuard.
            </p>

          </div>

          <div className="bg-white border border-slate-200 rounded-2xl p-5 md:p-8 shadow-lg">

            <label htmlFor="url-input" className="font-semibold block mb-3 text-slate-900">
              Website URL
            </label>

            <div className="flex flex-col sm:flex-row gap-3">

              <input
                id="url-input"
                type="text"
                value={url}
                onChange={(event) => {
                  setUrl(event.target.value);
                  setValid(true);
                }}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    handleScan();
                  }
                }}
                placeholder="https://example.com"
                disabled={loading}
                className="flex-1 min-w-0 bg-white border border-slate-300 rounded-xl px-4 py-3 text-slate-900 outline-none focus:border-blue-600 focus:ring-2 focus:ring-blue-100"
              />

              <button
                onClick={handleScan}
                disabled={loading}
                className="bg-blue-700 text-white px-7 py-3 rounded-xl font-bold flex items-center justify-center gap-2 hover:bg-blue-800 disabled:opacity-50 transition"
              >
                {loading ? (
                  <>
                    <span className="animate-spin">◌</span>
                    Scanning...
                  </>
                ) : (
                  <>
                    <Search size={18} />
                    Scan URL
                  </>
                )}
              </button>

            </div>

            {!valid && (
              <p className="text-red-600 text-sm mt-3">
                Please enter a valid website URL.
              </p>
            )}

            {backendDown && !mockMode && (
              <p className="text-amber-700 text-sm mt-3">
                Backend unavailable or detection models are not loaded.
              </p>
            )}

            <p className="text-xs text-slate-500 mt-4">
              Only submit URLs you are authorized to analyze.
              Scan results are automated estimates, not security guarantees.
            </p>

            {/* SCAN RESULT */}

            <AnimatePresence>
              {result && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 20 }}
                  className="mt-7 border border-slate-200 rounded-xl p-6 bg-slate-50"
                >

                  {result.error ? (
                    <div className="flex items-start gap-3">

                      <XCircle className="text-red-600 shrink-0" size={25} />

                      <div>
                        <h3 className="font-bold text-red-600">
                          Scan Failed
                        </h3>

                        <p className="text-slate-600 text-sm mt-2">
                          {result.message}
                        </p>
                      </div>

                    </div>
                  ) : (
                    <>

                      <div className="flex flex-wrap justify-between items-center gap-3">

                        <div className="flex items-center gap-3">

                          {result.verdict.toLowerCase() === 'phishing' ? (
                            <AlertTriangle className="text-red-600" size={28} />
                          ) : (
                            <CheckCircle className="text-green-600" size={28} />
                          )}

                          <div>
                            <h3 className="font-bold text-xl text-slate-900">
                              Detection Result
                            </h3>

                            <p className="text-sm text-slate-500">
                              Automated URL analysis
                            </p>
                          </div>

                        </div>

                        <span
                          className={`px-4 py-2 rounded-full font-bold text-sm ${
                            result.verdict.toLowerCase() === 'phishing'
                              ? 'bg-red-100 text-red-700'
                              : 'bg-green-100 text-green-700'
                          }`}
                        >
                          {result.verdict.toUpperCase()}
                        </span>

                      </div>

                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mt-7">

                        <div className="bg-white border border-slate-200 rounded-xl p-4">
                          <p className="text-sm text-slate-500">
                            Risk Score
                          </p>

                          <p className="text-2xl font-bold mt-1 text-slate-900">
                            {Math.round((Number(result.score) || 0) * 100)}%
                          </p>
                        </div>

                        <div className="bg-white border border-slate-200 rounded-xl p-4">
                          <p className="text-sm text-slate-500">
                            Analysis Stage
                          </p>

                          <p className="text-2xl font-bold mt-1 text-slate-900">
                            {result.stage === 1
                              ? 'Quick'
                              : result.stage === 2
                              ? 'Deep'
                              : 'N/A'}
                          </p>
                        </div>

                        <div className="bg-white border border-slate-200 rounded-xl p-4 col-span-2 sm:col-span-1">
                          <p className="text-sm text-slate-500">
                            Latency
                          </p>

                          <p className="text-2xl font-bold mt-1 text-slate-900">
                            {result.latency_ms ?? 0} ms
                          </p>
                        </div>

                      </div>

                      {result.reasons && result.reasons.length > 0 && (
                        <div className="mt-6">

                          <h4 className="font-bold mb-3 text-slate-900">
                            Reasons for this result
                          </h4>

                          <ul className="space-y-2">
                            {result.reasons.map((reason, index) => (
                              <li
                                key={index}
                                className="flex items-start gap-2 text-sm text-slate-600"
                              >
                                <AlertTriangle
                                  size={17}
                                  className="text-blue-700 shrink-0 mt-0.5"
                                />

                                <span>
                                  {reason.text}

                                  {reason.weight !== undefined &&
                                    ` (Weight: ${reason.weight})`}
                                </span>
                              </li>
                            ))}
                          </ul>

                        </div>
                      )}

                      {result.note && (
                        <p className="text-sm text-slate-600 mt-4">
                          {result.note}
                        </p>
                      )}

                      {mockMode && (
                        <p className="text-xs text-amber-700 mt-4">
                          Demo mode: this result is sample data, not a real scan.
                        </p>
                      )}

                    </>
                  )}

                </motion.div>
              )}
            </AnimatePresence>

          </div>
        </div>
      </section>

      {/* HOW IT WORKS AND LIMITATIONS */}

      <section className="px-5 md:px-10 py-16 bg-white">
        <div className="max-w-7xl mx-auto grid md:grid-cols-2 gap-8">

          <div className="bg-slate-50 border border-slate-200 rounded-2xl p-7">
            <h3 className="text-2xl font-bold flex items-center gap-3 text-slate-900">
              <Zap className="text-blue-700" />
              How PhishGuard Works
            </h3>

            <ol className="mt-6 space-y-4 text-slate-600">
              <li>01. Enter the website URL you want to analyze.</li>
              <li>02. The application sends the URL to the backend API.</li>
              <li>03. The detection system analyzes the submitted URL.</li>
              <li>04. The application displays a verdict, score and reasons.</li>
            </ol>
          </div>

          <div className="bg-slate-50 border border-slate-200 rounded-2xl p-7">
            <h3 className="text-2xl font-bold flex items-center gap-3 text-slate-900">
              <AlertTriangle className="text-blue-700" />
              Important Limitations
            </h3>

            <ul className="mt-6 space-y-4 text-slate-600">
              <li>• Automated predictions may contain false positives or false negatives.</li>
              <li>• Newly created or targeted phishing websites may not be detected.</li>
              <li>• A safe-looking result does not guarantee that a website is trustworthy.</li>
              <li>• Always verify sensitive requests through official channels.</li>
            </ul>
          </div>

        </div>
      </section>

      {/* SESSION STATISTICS */}

      <section className="bg-slate-50 border-y border-slate-200 px-5 py-12">
        <div className="max-w-5xl mx-auto">

          <h2 className="text-2xl font-bold text-center mb-9 text-slate-900">
            Your Scanning Activity
          </h2>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 text-center">

            <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
              <Search className="mx-auto text-blue-700 mb-3" size={28} />

              <p className="text-3xl font-extrabold text-slate-900">
                {formatNumber(session.scans)}
              </p>

              <p className="text-slate-600 mt-2">
                Scans this session
              </p>
            </div>

            <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
              <AlertTriangle className="mx-auto text-red-600 mb-3" size={28} />

              <p className="text-3xl font-extrabold text-slate-900">
                {formatNumber(session.phishing)}
              </p>

              <p className="text-slate-600 mt-2">
                Flagged as phishing
              </p>
            </div>

            <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
              <Zap className="mx-auto text-blue-700 mb-3" size={28} />

              <p className="text-3xl font-extrabold text-slate-900">
                {formatNumber(avgLatency)} ms
              </p>

              <p className="text-slate-600 mt-2">
                Average response time
              </p>
            </div>

          </div>
        </div>
      </section>

      {/* RECENT SCANS */}

      {recent.length > 0 && (
        <section className="px-5 py-10 bg-white">
          <div className="max-w-4xl mx-auto">

            <div className="flex justify-between items-center mb-5">
              <h2 className="text-2xl font-bold text-slate-900">
                Recent Scans
              </h2>

              <button
                onClick={() => setRecent([])}
                className="text-blue-700 underline text-sm"
              >
                Clear History
              </button>
            </div>

            <div className="space-y-3">
              {recent.map((scan, index) => (
                <div
                  key={`${scan.url}-${index}`}
                  className="bg-slate-50 border border-slate-200 p-4 rounded-xl flex flex-wrap justify-between gap-3 items-center"
                >

                  <div className="flex items-center gap-3 min-w-0">
                    <Globe className="text-blue-700 shrink-0" size={20} />

                    <span className="truncate text-slate-800">
                      {(() => {
                        try {
                          return new URL(scan.url).hostname;
                        } catch {
                          return scan.url;
                        }
                      })()}
                    </span>
                  </div>

                  <span
                    className={`font-bold text-sm ${
                      scan.verdict.toLowerCase() === 'phishing'
                        ? 'text-red-600'
                        : 'text-green-600'
                    }`}
                  >
                    {scan.verdict.toUpperCase()}
                  </span>

                </div>
              ))}
            </div>

          </div>
        </section>
      )}

      {/* FOOTER */}

      <footer className="bg-slate-100 text-slate-700 border-t border-slate-200 px-5 py-12">
        <div className="max-w-7xl mx-auto">

          <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-10">

            <div>
              <div className="flex items-center gap-2 text-slate-900">
                <ShieldCheck size={26} />

                <span className="text-2xl font-extrabold">
                  PhishGuard
                </span>
              </div>

              <p className="text-sm text-slate-600 mt-4 leading-relaxed">
                An academic phishing detection project designed to
                help users recognize suspicious URLs and understand
                online security threats.
              </p>
            </div>

            <div>
              <h3 className="font-bold text-slate-900 mb-4">
                Quick Links
              </h3>

              <div className="flex flex-col gap-3 text-sm">
                <a href="#about" className="hover:text-blue-700">
                  About Phishing
                </a>

                <a href="#statistics" className="hover:text-blue-700">
                  Threat Statistics
                </a>

                <a href="#types" className="hover:text-blue-700">
                  Attack Types
                </a>

                <a href="#protection" className="hover:text-blue-700">
                  Safety Tips
                </a>
              </div>
            </div>

            <div>
              <h3 className="font-bold text-slate-900 mb-4">
                Try PhishGuard
              </h3>

              <p className="text-sm text-slate-600 mb-4">
                Analyze a suspicious URL using our detection application.
              </p>

              <button
                onClick={scrollToScanner}
                className="bg-blue-700 text-white px-5 py-2.5 rounded-lg font-semibold hover:bg-blue-800 transition"
              >
                Scan a URL →
              </button>
            </div>

          </div>

          <div className="border-t border-slate-200 mt-10 pt-6 text-center text-xs text-slate-500">
            <p>© {new Date().getFullYear()} PhishGuard. Built for academic purposes.</p>

            <p className="mt-2">
              Automated detection is advisory and is not a guarantee of website safety.
            </p>
          </div>

        </div>
      </footer>

    </div>
  );
}
