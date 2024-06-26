+++
author = "Laychiva Chhout"
title = "បញ្ញាសិប្បនិម្មិត - Artificial Intelligence និង ChatGPT"
date = "2024-06-10"
description = "ការឈ្វេងយល់ពីបញ្ញាសិប្បនិម្មិត និង ChatGPT"
math = "true"
tags = [
    "ai"
]
categories = [
    "Artificial Intelligence"
]
series = ["Themes Guide"]
aliases = ["migrate-from-jekyl"]
image = "cover.png"
draft= "false"
+++

## សេចក្តីផ្តើម

នៅក្នុងសង្គមយើងសព្វថ្ងៃ AI ត្រូវបានប្រើប្រាស់យ៉ាងទូលំទូលាយជាពិសេស ChatGPT ដែលយើងតែងតែប្រើប្រាស់ដើម្បីសួរសំណួរ បកប្រែឯកសារ សង្ខេបអត្ថបទជាដើម។ តែយើងមិនបានដឹងនោះទេថាតើវាជាអ្វីឱ្យពិតប្រាកដនោះទេ តើយើងត្រូវការអ្វីខ្លះដើម្បីបង្កើតកម្មវិធីមួយដូច ChatGPT នេះបាន? 
តាមពិតទៅមុននឹងឈានដល់ភាពល្បីល្បាញ ហើយត្រូវបានពិភាក្សាយ៉ាងទូលំទូលាយដូចសព្វថ្ងៃនេះ AI ត្រូវបានប្រើប្រាស់យ៉ាងច្រើនរួចទៅហើយ នៅក្នុងវិស័យផ្សេងៗជាច្រើន និងឧបករណ៍ប្រើប្រាស់ប្រចាំថ្ងៃជាច្រើន ដែលយើងបានប្រើប្រាស់តែមិនបានចាប់អារម្មណ៍ដូចជា Siri នៅក្នុងទូរសព្ទដៃ, ម៉ាស៊ីនណែនាំនៅក្នុងបណ្តាញសង្គមផ្សេងៗ (Recommendation Engine),  ម៉ាស៊ីនស្វែងរក (Search Engine), និងការប្រើប្រាស់ជាច្រើនទៀតនៅក្នុងវិស័យឧស្សាហកម្ម។ តែ ChatGPT នេះមានលក្ខណៈពិសេសជាង AI ផ្សេងៗដែលមានពីមុនមកនោះ គឺសមត្ថភាពក្នុងការឆ្លើយតបរបស់វា ដែលមានលក្ខណៈស្រដៀង ឬពូកែជាងមនុស្សផងដែរ។  ក្នុងអត្ថបទនេះខ្ញុំនឹងបកស្រាយដោយងាយយល់ពី ប្រវត្តិ ការបង្កើត ការដំណើរការរបស់ ChatGPT នេះ។ 

## ប្រវត្តិរបស់ ChatGPT
មុននឹងស្វែងយល់ពី ChatGPT យើងត្រូវតែដឹងថាអ្វីដែលស្ថិតនៅពីក្រោយកម្មវិធីមួយនេះ។ ChatGPT គឺជាកម្មវិធីមួយដែលប្រើប្រាស់នូវម៉ូដែលដែលបង្កើតឡើងដោយក្រុមហ៊ុន OpenAI ដែលសព្វថ្ងៃនេះមានម៉ូដែលជាច្រើនដែលកំពុងត្រូវបានប្រើប្រាស់ដូចជា GPT-3.5 GPT-3.5-Turbo, GPT-4 និង GPT-4o ជាដើម។ មុននឹងការមកដល់របស់ម៉ូដែលទាំងអស់នេះក្រុមហ៊ុន OpenAI បានធ្វើការស្រាវជ្រាវ និងចេញនូវម៉ូដែលផ្សេងៗជាច្រើន និងធ្វើការអភិវឌ្ឍជាបន្តបន្ទាប់រហូតទទួលបានលទ្ធផលដូចសព្វថ្ងៃនេះ ។  ខ្ញុំនឹងរៀបរាប់ពីប្រវត្តិរបស់វាដោយសង្ខេបដូចខាងក្រោម៖

### ឆ្នាំ២០១៧

នៅក្នុងឆ្នាំ ២០១៧ អ្នកវិទ្យាសាស្រ្តមួយក្រុមដែលដឹកនាំដោយលោក Ashish Vaswani បានធ្វើការស្រាវជ្រាវ និង បង្ហោះជាសាធារណៈនូវឯកសារស្រាវជ្រាវមួយដែលមានឈ្មោះថា “Attention is All You Need” ដែលឯកសារនេះមានសារៈសំខាន់បំផុតក្នុងការធ្វើឱ្យមានភាពជឿនលឿនក្នុងវិស័យនេះដូចសព្វថ្ងៃ។ តើហេតុអ្វីបានជាវាមានសារៈសំខាន់បែបនេះ ? នៅក្នុងឯកសារស្រាវជ្រាវនេះអ្នកស្រាវជ្រាវបាននាំយកគំនិតថ្មីមួយនោះគឺ Attention និង Transformer។ 

#### តើអ្វីទៅជា Attention និង Transformer ?
មុននឹងបកស្រាយយើងត្រឡប់ទៅមើលស្ថានភាពមុនការរកឃើញរបស់វា តើ AI ជួបបញ្ហាអ្វី ? នៅពេលនោះ ក្នុងអេអាយ ឬ ក្នុងការសិក្សាភាសាធម្មជាតិ (Natural Language Processing) ជាពិសេសគឺ ការដោះស្រាយបញ្ហាណាដែលតម្រូវឱ្យមានការបង្កើតអក្សរ (Text Generation), អ្នកវិទ្យាសាស្ត្រ ឬ អ្នកអភិវឌ្ឍន៍កម្មវិធីអេអាយទាំងអស់ជួបបញ្ហាមួយដែលពិបាកក្នុងការដោះស្រាយ នោះគឺប្រវែងរបស់ធាតុចូល (Input Length)។ ដើម្បីឱ្យងាយស្រួលយល់ ឧទាហរណ៍ថាយើងចង់បកប្រែល្បះ “The bank of the river” និង “Money in the bank” នៅពេលដែលយើងអានល្បះនីមួយៗចប់យើងដឹងភ្លាមថាតើ Bank នៅក្នុងល្បះទីមួយ និង នៅក្នុងល្បះទីពីរ មានអត្ថន័យបែបណា។ នេះដោយសារតែយើងបានអានចប់នូវល្បះនីមួយៗដែលក្នុងនោះមានពាក្យផ្សេងទៀតដែលអាចផ្តល់នូវព័ត៌មានមួយដែលយើងអាចប្រើប្រាស់ដើម្បីបកប្រែល្បះនេះឱ្យមានភាពត្រឹមត្រូវ។ ដូចមនុស្សយើងដែរ អេអាយ ក៏ត្រូវការព័ត៌មានទាំងអស់នេះដូចគ្នាដើម្បីធ្វើការបកប្រែល្បះនីមួយៗឱ្យមានភាពត្រឹមត្រូវ។ ចុះបើសិនជាល្បះដែលត្រូវបកប្រែនេះមានប្រវែងវែងនោះតើនិងមានបញ្ហាអ្វីកើតឡើង ? មនុស្សយើងមានលក្ខណៈពិសេសមួយក្នុងការស្វែងរកពាក្យដែលអាចផ្តល់ន័យបន្ថែម ហើយធ្វើការបកប្រែមួយដែលមានភាពត្រឹមត្រូវ ផ្ទុយមកវិញនៅពេលនោះ អេអាយ មិនមានសមត្ថភាពបែបនេះទេពោលគឺវាត្រូវស្វែងយល់ និងចងចាំនូវអត្ថបទទាំងស្រុងហើយប្រើប្រាស់ការចងចាំនេះដើម្បីបកប្រែ។ នៅពេលដែលល្បះ ឬ អត្ថបទមានប្រវែងវែងនោះ អេអាយ មានការលំបាកក្នុងការចង់ចាំ និងធ្វើឱ្យបាត់បង់នូវព័ត៌មានមួយចំនួនដែលជាហេតុធ្វើឱ្យសមត្ថភាពរបស់វាមានកម្រិត។ នៅពេលនោះហើយដែល Attention បានចូលមក ហើយដោះស្រាយបញ្ហានេះ ដោយ Attention ដំណើរការដូចទៅនឹងមនុស្សដែរពោលគឺស្វែងរក និងប្រើប្រាស់តែព័ត៌មានណាដែលសំខាន់ និងត្រូវការតែប៉ុណ្ណោះ មិនមែនចងចាំនូវអត្ថបទទាំងស្រុងនោះទេ។ 
Attention នេះត្រូវបានក្នុងទម្រង់មួយដែលគេឱ្យឈ្មោះថា Transformer។ Transformer នេះផ្នែកសំខាន់ៗពីរគឺ ៖ Encoder និង Decoder។ ខ្ញុំនឹងមិនពន្យល់ដោយប្រើប្រាស់ពាក្យបច្ចេកទេសនោះទេដើម្បីឱ្យអ្នកទាំងអស់គ្នាងាយស្រួលយល់ ដោយក្នុងនេះយើងអាចគិតថា Encoder និង Decoder នេះជាក្រុមមនុស្សពីរផ្សេងគ្នាដែលធ្វើការជាមួយគ្នាក្នុងការដោះស្រាយបញ្ហាអ្វីមួយ។ ឧទាហរណ៍ថាយើងចង់បកប្រែឯកសារមួយពីភាសា ក ទៅភាសា ខ នោះក្រុមទាំងពីរនោះនឹងធ្វើការដូចតទៅ៖
ក្រុមទីមួយ Encoder ពួកគេនឹងអានឯកសារនោះមួយចប់ហើយពួកគេនឹងមិនធ្វើការបកប្រែភ្លាមៗនោះទេ។ ពួកគេផ្តោតទៅលើការស្រង់យកនូវពាក្យ ឃ្លា និងទម្រង់ សំខាន់ៗដែលអាចធ្វើឱ្យយើងយល់ពីអត្ថន័យ និងមូលន័យរបស់ឯកសារនោះ។ នៅក្នុងក្រុមនេះ អ្នកបកប្រែនីមួយៗធ្វើការផ្សេងគ្នាលើផ្នែកផ្សេងៗរបស់ឯកសារ ដោយខ្លះធ្វើការលើរបៀបនៃការសរសេរ ការប្រើប្រាស់ពាក្យ ហើយខ្លះទៀតធ្វើការលើមនោសញ្ចេតនាជាដើម។ ពួកគេនឹងយករបស់ដែលពួកគេស្វែងយល់នោះដាក់ចូលគ្នាហើយបង្កើតជាផែនទីមួយដែលអាចឱ្យក្រុមបកប្រែមួយទៀតយល់បាន។ 
បន្ទាប់មកក្រុមមួយទៀត(Decoder)នឹងប្រើប្រាស់ផែនទីនេះដើម្បីធ្វើការបកប្រែទៅភាសា ខ។ ការធ្វើបែបនេះក្រុម Decoder នឹងមិនធ្វើការបកបកប្រែម្តងមួយពាក្យៗនោះទេ តែពួកគេបានប្រើប្រាស់នូវព័ត៌មានសំខាន់ៗទាំងអស់ដែលបានមកពីក្នុង Encoder ។ 
### ឆ្នាំ ២០១៨
នៅឆ្នាំ ២០១៨ OpenAI បានបញ្ចេញម៉ូដែលមួយដែលឱ្យឈ្មោះថា GPT-1 ដោយម៉ូដែលនេះមានសមត្ថភាពក្នុងការបង្កើតពាក្យ (Generate text) ដោយសារតែម៉ូដែលនេះប្រើប្រាស់តែផ្នែកទី២ របស់ Transformer តែប៉ុណ្នោះ ពោលគឺ Decoder។ GPT-1 នេះត្រូវបានបង្ហាត់ដោយប្រើប្រាស់ Unsupervised learning ជាមួយនឹងទិន្នន័យដ៏ច្រើនមហិមាដែលធ្វើឱ្យ Decoder នេះមានសមត្ថភាពក្នុងការទាយនូវពាក្យនៅក្នុងល្បះដោយមានពាក្យមុនៗនៅក្នុងដៃ ( predict the next word in a sequence given all the previous words) ។ GPT-1 នេះមានចំនួនប៉ារ៉ាមែត្រ ១១៧ លាន។ 
#### តើអ្វីទៅជាការបង្វឹកម៉ូដែល ? 
ការបង្វឹកម៉ូដែលអេអាយមាន៣របៀបធំៗដូចជា Supervised Learning, Unsupervised Learning និង Reinforcement Learning តែនៅក្នុងអត្ថបទខ្លីនេះ ខ្ញុំនឹងនិយាយន័យជារួមមួយដើម្បីឱ្យមិត្តអ្នកអានអាចយល់ថាអ្វីជាការបង្ហាត់ម៉ូដែល។ ការបង្ហាត់នីមួយៗមានលក្ខណៈខុសគ្នាមែន តែគោលដៅរបស់ការបង្ហាត់នេះគឺធ្វើយ៉ាងណាឱ្យម៉ូដែលនោះធ្វើការទស្សន៍ទាយមានកំហុសតិចបំផុត (Minimise Loss Function) ។ ដើម្បីសម្រេចបានវត្ថុបំណងនេះគេត្រូវផ្តល់ទិន្នន័យចាំបាច់ដែលបម្រើឱ្យការទស្សន៍ទាយនេះហើយភាគច្រើនទិន្នន័យទាំងនោះតែងតែមានទំហំធំ។ ឧទាហរណ៍ថាយើងចង់បង្ហាត់ម៉ូដែលមួយឱ្យចេះមើលថាតើរូបសត្វដែលយើងផ្តល់ឱ្យជាសត្វឆ្កែ ឬសត្វឆ្មា នោះយើងត្រូវផ្តល់ទិន្នន័យជាច្រើនដែលក្នុងនោះមានរូបសត្វឆ្កែ និងសត្វឆ្មា ហើយរូបនីមួយៗត្រូវតែមានស្លាកសញ្ញាបញ្ជាក់ថាជាសត្វអ្វី។ 
បន្ថែមពីលើការបង្វឹកម៉ូដែលទូទៅនេះ ចំពោះការបង្ហាត់ម៉ូដែល Transformer វាមានលក្ខណៈស្រដៀងគ្នានឹងការពន្យល់ខាងលើដែរ ដោយការបង្ហាត់នោះធ្វើយ៉ាងណាឱ្យការទស្សន៍ទាយមានកំហុសតិចបំផុត តាមរយៈការផ្លាស់ប្តូរប៉ារ៉ាមែត្ររបស់ម៉ូដែល (Parameters update)។ 
#### តើអ្វីទៅជាប៉ារ៉ាមែត្រ ? 
ប៉ារ៉ាមែត្រនេះមិនមែនជាចំនួនទិន្នន័យនោះទេហើយ ការសន្និដ្ឋានថាចំនួនប៉ារ៉ាមែត្រច្រើនបង្ហាញថាម៉ូដែលបានបង្ហាត់ជាមួយទិន្នន័យច្រើនក៏មិនពិតនោះដែរ។ ប៉ារ៉ាមែត្រ ជាអញ្ញាតមួយដែលត្រូវបានប្រើប្រាស់ដើម្បីកំណត់នូវទម្រង់ ឬអកប្បកិរិយារបស់ម៉ូដែលមួយ ដោយម៉ូដែលដែលមានចំនួនប៉ារ៉ាមែត្រច្រើនអាចធ្វើការស្វែងយល់ពីទិន្នន័យដែលមានសភាពស្មុគស្មាញបានល្អជាងម៉ូដែលដែលមានប៉ារ៉ាមែត្រតិច។ 
### ឆ្នាំ២០១៩ 
OpenAI បានបញ្ចេញនូវម៉ូដែលរបស់ខ្លួនមួយទៀតដែលមានសមត្ថភាពខ្លាំងជាងម៉ូដែលរបស់ខ្លួននោះគឺ GPI-2 ដែលមានចំនួនប៉ារ៉ាមែត្រសរុប ១៥០០លាន ។ ដែលនៅពេលនោះម៉ូដែលនេះត្រូវបានគេប្រើប្រាស់ដើម្បីបង្កើតព័ត៌មានក្លែងក្លាយ និងធ្វើឱ្យមានការបារម្ភពីការប្រើប្រាស់របស់វាដោយសារតែការបង្កើតពាក្យ ឬល្បះមានភាពដូចទៅហ្នឹងមនុស្សខ្លាំងពេក។ 
### ឆ្នាំ២០២០
OpenAI បានបង្កើនទំហំទិន្នន័យដែលប្រើប្រាស់ដើម្បីបង្ហាត់ម៉ូដែលមុនៗ ដោយបញ្ចូលដូចជា សៀវភៅ ឯកសារផ្សេងៗលើអុីនធឺណែតជាដើម ហើយពួកគេបានបង្ហាត់ម៉ូដែលថ្មីមួយទៀតដែលមានឈ្មោះថា GPT-3 ហើយមានចំនួនប៉ារ៉ាម៉ែត្រសរុបចំនួន ១៧៥ ពាន់លាន។ 
### ឆ្នាំ២០២២ 
OpenAI បានដាក់ឱ្យប្រើប្រាស់ជាផ្លូវការនូវ ChatGPT ដែលប្រើប្រាស់ម៉ូដែល GPT-3.5-Turbo ដែលម៉ូដែលនេះត្រូវបានបង្ហាត់ដោយប្រើប្រាស់ GPT-3 និង ទិន្នន័យបែបពិភាក្សា (Conversational Data) ដែលធ្វើឱ្យម៉ូដែលនេះអាចឆ្លើយឆ្លងជាមួយនឹងមនុស្សបាន។ 
### ឆ្នាំ២០២៣
ម៉ូដែលដែលមានសមត្ថភាពខ្លាំងបំផុតរបស់ OpenAI ត្រូវបានដាក់ឱ្យអ្នកដែលបានចុះឈ្មោះបង់ប្រាក់ប្រើប្រាស់ ហើយ OpenAI មិនបានបញ្ជាក់ថាម៉ូដែលនេះមានចំនួនប៉ារ៉ាមែត្រប៉ុន្មាននោះទេ តែគេជឿថាមានចំនួន ១,៧៦ ពាន់ពាន់លាន។ 
ការចំណាយលើការបង្វឹក
មកដល់ចំណុចនេះយើងអាចដឹងថាការបង្ហាត់ម៉ូដែលនេះមិនមែនជារឿងងាយស្រួលនោះទេ ពោលគឺយើងត្រូវការទិន្នន័យដ៏ច្រើនមហិមារ ហើយទិន្នន័យទាំងនោះមិនមែនងាយស្រួលក្នុងការស្វែងរកនោះទេ មានតែក្រុមហ៊ុនធំៗដែលមានសមត្ថភាពគ្រប់គ្រាន់ក្នុងការស្វែងរក និង មានការអនុញ្ញាត្តិចូលទៅកាន់ទិន្នន័យទាំងអស់នោះតែប៉ុណ្នោះដែលអាចធ្វើទៅបាន។ បន្ថែមពីទិន្នន័យនេះទៅទៀតការ បង្ហាត់ម៉ូដែលនីមួយៗត្រូវការ ការចំណាយប្រាក់ច្រើនសន្ធឹកសន្ធាប់ណាស់ដោយសារតែម៉ូដែលទាំងនោះមានចំនួនប៉ារ៉ាមែត្រច្រើន ដែលធ្វើឱ្យមានការចាំបាច់ក្នុងការប្រើប្រាស់ម៉ាស៊ីនដែលមានកម្លាំងគណនាខ្លាំងដើម្បីបង្កាត់។ ឧទាហរណ៍ ម៉ូដែល GPT-3 ត្រូវការគណនា ៣,១៤ ស្វ័យគុណ ២៣ (3,14 trillion trillion) ដង។ ដើម្បីគណនាចេញជាប្រាក់នោះយើងអាចគណនាបានដូចតទៅ ៖ បើសិនជាការបង្ហាត់នេះប្រើប្រាស់ GPT V100 ដែលមានតម្លៃប្រហែល ១០ ០០០ដុល្លារនោះ ហើយ H100 អាចគណនាបាន ១៤ Trillion ក្នុងមួយវិនាទីនោះ យើងត្រូវការពេលប្រហែល ៧១១ ឆ្នាំដើម្បីបង្ហាត់ម៉ូដែលនោះ។ OpenAI បានប្រកាសថាគេបានប្រើប្រាស់ប្រាក់អស់ប្រហែល ៤,៦ លានដុល្លារស្រដែងដើម្បីបង្កាត់ម៉ូដែលមួយនេះ។ យើងអាចស្មានបានថាបើ GPT-4 មានចំនួនប៉ារ៉ាមែត្ររហូតដល់ទៅ ១,៧៦ ពាន់ពាន់លាន នោះយើងត្រូវការប្រាក់ប៉ុន្មានដើម្បីបង្ហាត់វា។ 
## សេចក្តីសន្និដ្ឋាន
ក្នុងអត្ថបទនេះខ្ញុំគ្រាន់តែចង់ឱ្យអ្នកទាំងអស់គ្នាយល់ថាអ្វីជា ChatGPT ហើយការបង្ហាត់របស់វាត្រូវការអ្វីខ្លះតែប៉ុណ្នោះ។ ជាមួយនឹងតម្រូវការបច្ចេកទេស និងប្រាក់ទាំងអស់នេះ ខ្ញុំគិតថាមិនមែនក្រុមហ៊ុនមួយ ឬមនុស្សគ្រប់គ្នាអាចមានសមត្ថភាពក្នុងការបង្ហាត់វាបាននោះទេ យើងអាចត្រឹមប្រើប្រាស់ ម៉ូដែលរបស់គេដែលមានស្រាប់តែប៉ុណ្ណោះ។ 


## ឯកសារយោង
https://blog.wordbot.io/ai-artificial-intelligence/a-brief-history-of-the-generative-pre-trained-transformer-gpt-language-models/



