#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace std;

namespace fs = std::filesystem;

const int PAGE_SIZE = 100;
const int REQUEST_DELAY_MS = 150;
const double MB = 1048576.0;

#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#endif

// runs a shell command and returns its output
string run_command(const string& command)
{
    array<char, 8192> buffer{};
    string result;

    FILE* pipe = popen(command.c_str(), "r");

    if (!pipe)
    {
        throw runtime_error("could not run command");
    }

    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr)
    {
        result += buffer.data();
    }

    int code = pclose(pipe);

    if (code != 0)
    {
        throw runtime_error("command failed: " + command);
    }

    return result;
}

// converts text into a safe url value
string url_encode(const string& value)
{
    ostringstream encoded;

    for (unsigned char c : value)
    {
        if (
            isalnum(c) ||
            c == '-' ||
            c == '_' ||
            c == '.' ||
            c == '~'
        )
        {
            encoded << c;
        }
        else
        {
            encoded << '%';
            encoded << "0123456789ABCDEF"[c >> 4];
            encoded << "0123456789ABCDEF"[c & 15];
        }
    }

    return encoded.str();
}

// downloads text from a url with retries
string http_get(const string& url)
{
    const int max_attempts = 5;

    for (int attempt = 1; attempt <= max_attempts; attempt++)
    {
        try
        {
            string command =
                "curl -L -sS --fail "
                "--connect-timeout 20 "
                "--max-time 120 "
                "\"" + url + "\"";

            return run_command(command);
        }
        catch (const exception& error)
        {
            cout << "\nrequest failed attempt "
                 << attempt
                 << " of "
                 << max_attempts
                 << endl;

            if (attempt == max_attempts)
            {
                throw;
            }

            int wait_seconds = attempt * 3;
            this_thread::sleep_for(chrono::seconds(wait_seconds));
        }
    }

    throw runtime_error("request failed");
}

// prepares a value for safe csv writing
string csv_escape(const string& value)
{
    bool needs_quotes = false;

    for (char c : value)
    {
        if (c == ',' || c == '"' || c == '\n' || c == '\r')
        {
            needs_quotes = true;
            break;
        }
    }

    if (!needs_quotes)
    {
        return value;
    }

    string result = "\"";

    for (char c : value)
    {
        if (c == '"')
        {
            result += "\"\"";
        }
        else
        {
            result += c;
        }
    }

    result += "\"";
    return result;
}

// converts a hex character into a number
int hex_value(char c)
{
    if (c >= '0' && c <= '9')
    {
        return c - '0';
    }

    if (c >= 'a' && c <= 'f')
    {
        return 10 + c - 'a';
    }

    if (c >= 'A' && c <= 'F')
    {
        return 10 + c - 'A';
    }

    return -1;
}

// reads four hex characters into a number
bool parse_hex4(const string& text, size_t pos, int& value)
{
    if (pos + 4 > text.size())
    {
        return false;
    }

    value = 0;

    for (int i = 0; i < 4; i++)
    {
        int h = hex_value(text[pos + i]);

        if (h < 0)
        {
            return false;
        }

        value = value * 16 + h;
    }

    return true;
}

// appends a unicode codepoint as bytes
void append_utf8(string& out, int codepoint)
{
    if (codepoint <= 0x7F)
    {
        out += static_cast<char>(codepoint);
    }
    else if (codepoint <= 0x7FF)
    {
        out += static_cast<char>(0xC0 | ((codepoint >> 6) & 0x1F));
        out += static_cast<char>(0x80 | (codepoint & 0x3F));
    }
    else if (codepoint <= 0xFFFF)
    {
        out += static_cast<char>(0xE0 | ((codepoint >> 12) & 0x0F));
        out += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (codepoint & 0x3F));
    }
    else
    {
        out += static_cast<char>(0xF0 | ((codepoint >> 18) & 0x07));
        out += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (codepoint & 0x3F));
    }
}

// converts escaped json text into normal text
string json_unescape(const string& value)
{
    string result;

    for (size_t i = 0; i < value.size(); i++)
    {
        char c = value[i];

        if (c != '\\')
        {
            result += c;
            continue;
        }

        if (i + 1 >= value.size())
        {
            break;
        }

        char next = value[++i];

        if (next == 'n')
        {
            result += '\n';
        }
        else if (next == 'r')
        {
            result += '\r';
        }
        else if (next == 't')
        {
            result += '\t';
        }
        else if (next == 'b')
        {
            result += '\b';
        }
        else if (next == 'f')
        {
            result += '\f';
        }
        else if (next == '"')
        {
            result += '"';
        }
        else if (next == '\\')
        {
            result += '\\';
        }
        else if (next == '/')
        {
            result += '/';
        }
        else if (next == 'u')
        {
            int codepoint = 0;

            if (!parse_hex4(value, i + 1, codepoint))
            {
                continue;
            }

            i += 4;

            if (codepoint >= 0xD800 && codepoint <= 0xDBFF)
            {
                if (
                    i + 6 < value.size() &&
                    value[i + 1] == '\\' &&
                    value[i + 2] == 'u'
                )
                {
                    int low = 0;

                    if (parse_hex4(value, i + 3, low))
                    {
                        if (low >= 0xDC00 && low <= 0xDFFF)
                        {
                            codepoint =
                                0x10000 +
                                ((codepoint - 0xD800) << 10) +
                                (low - 0xDC00);

                            i += 6;
                        }
                    }
                }
            }

            append_utf8(result, codepoint);
        }
        else
        {
            result += next;
        }
    }

    return result;
}

// finds the closing brace for a json object
size_t find_matching_brace(const string& text, size_t open_pos)
{
    bool in_string = false;
    bool escaped = false;
    int depth = 0;

    for (size_t i = open_pos; i < text.size(); i++)
    {
        char c = text[i];

        if (escaped)
        {
            escaped = false;
            continue;
        }

        if (c == '\\')
        {
            escaped = true;
            continue;
        }

        if (c == '"')
        {
            in_string = !in_string;
            continue;
        }

        if (in_string)
        {
            continue;
        }

        if (c == '{')
        {
            depth++;
        }
        else if (c == '}')
        {
            depth--;

            if (depth == 0)
            {
                return i;
            }
        }
    }

    return string::npos;
}

// extracts row objects from the downloaded json
vector<string> extract_row_objects(const string& json_text)
{
    vector<string> rows;
    size_t pos = 0;

    while (true)
    {
        pos = json_text.find("\"row\"", pos);

        if (pos == string::npos)
        {
            break;
        }

        size_t colon = json_text.find(':', pos);

        if (colon == string::npos)
        {
            break;
        }

        size_t open_brace = json_text.find('{', colon);

        if (open_brace == string::npos)
        {
            break;
        }

        size_t close_brace = find_matching_brace(json_text, open_brace);

        if (close_brace == string::npos)
        {
            break;
        }

        rows.push_back(json_text.substr(open_brace, close_brace - open_brace + 1));
        pos = close_brace + 1;
    }

    return rows;
}

// gets a value from a json object by key
string get_json_value(const string& object, const string& key)
{
    string pattern = "\"" + key + "\"";
    size_t key_pos = object.find(pattern);

    if (key_pos == string::npos)
    {
        return "";
    }

    size_t colon = object.find(':', key_pos);

    if (colon == string::npos)
    {
        return "";
    }

    size_t pos = colon + 1;

    while (pos < object.size() && isspace(static_cast<unsigned char>(object[pos])))
    {
        pos++;
    }

    if (pos >= object.size())
    {
        return "";
    }

    if (object[pos] == '"')
    {
        pos++;
        string value;
        bool escaped = false;

        while (pos < object.size())
        {
            char c = object[pos++];

            if (escaped)
            {
                value += '\\';
                value += c;
                escaped = false;
                continue;
            }

            if (c == '\\')
            {
                escaped = true;
                continue;
            }

            if (c == '"')
            {
                break;
            }

            value += c;
        }

        return json_unescape(value);
    }

    size_t end = pos;

    while (
        end < object.size() &&
        object[end] != ',' &&
        object[end] != '}'
    )
    {
        end++;
    }

    string value = object.substr(pos, end - pos);

    while (!value.empty() && isspace(static_cast<unsigned char>(value.front())))
    {
        value.erase(value.begin());
    }

    while (!value.empty() && isspace(static_cast<unsigned char>(value.back())))
    {
        value.pop_back();
    }

    if (value == "null")
    {
        return "";
    }

    return value;
}

// converts text into a double value
double to_double(const string& value)
{
    try
    {
        return stod(value);
    }
    catch (...)
    {
        return 0.0;
    }
}

// counts rows in a csv file
long long count_csv_rows(const fs::path& path)
{
    ifstream file(path);

    if (!file.is_open())
    {
        return 0;
    }

    long long lines = 0;
    string line;

    while (getline(file, line))
    {
        lines++;
    }

    if (lines == 0)
    {
        return 0;
    }

    return lines - 1;
}

// prints information about a saved file
void print_saved(const fs::path& path)
{
    long long rows = count_csv_rows(path);
    double size_mb = static_cast<double>(fs::file_size(path)) / MB;

    cout << "Saved -> "
         << path.filename().string()
         << " ("
         << rows
         << " rows, "
         << size_mb
         << " MB)"
         << endl;
}

// builds a dataset rows api url
string rows_url(
    const string& dataset,
    const string& config,
    const string& split,
    int offset
)
{
    ostringstream url;

    url << "https://datasets-server.huggingface.co/rows"
        << "?dataset=" << url_encode(dataset)
        << "&config=" << url_encode(config)
        << "&split=" << url_encode(split)
        << "&offset=" << offset
        << "&length=" << PAGE_SIZE;

    return url.str();
}

// writes the first row of a csv file
void write_csv_header(ofstream& out, const vector<string>& columns)
{
    for (size_t i = 0; i < columns.size(); i++)
    {
        if (i > 0)
        {
            out << ",";
        }

        out << columns[i];
    }

    out << "\n";
}

// writes one row into a csv file
void write_csv_row(ofstream& out, const vector<string>& values)
{
    for (size_t i = 0; i < values.size(); i++)
    {
        if (i > 0)
        {
            out << ",";
        }

        out << csv_escape(values[i]);
    }

    out << "\n";
}

// checks if the row limit was reached
bool should_stop(int saved, int max_rows)
{
    return max_rows > 0 && saved >= max_rows;
}

// shows the current download progress
void progress_line(const string& name, int saved, int offset)
{
    cout << "\r"
         << name
         << " | rows saved: "
         << saved
         << " | current offset: "
         << offset
         << flush;
}

// downloads and saves the jigsaw dataset
void fetch_jigsaw(const fs::path& out_dir, int max_rows)
{
    cout << "\nfetching jigsaw" << endl;

    fs::path output = out_dir / "jigsaw.csv";
    ofstream out(output);

    if (!out.is_open())
    {
        throw runtime_error("could not create " + output.string());
    }

    write_csv_header(out, {
        "comment_text",
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate"
    });

    int saved = 0;

    for (int offset = 0; ; offset += PAGE_SIZE)
    {
        if (should_stop(saved, max_rows))
        {
            break;
        }

        try
        {
            string url = rows_url(
                "thesofakillers/jigsaw-toxic-comment-classification-challenge",
                "default",
                "train",
                offset
            );

            string body = http_get(url);
            vector<string> rows = extract_row_objects(body);

            if (rows.empty())
            {
                break;
            }

            for (const string& row : rows)
            {
                if (should_stop(saved, max_rows))
                {
                    break;
                }

                write_csv_row(out, {
                    get_json_value(row, "comment_text"),
                    get_json_value(row, "toxic"),
                    get_json_value(row, "severe_toxic"),
                    get_json_value(row, "obscene"),
                    get_json_value(row, "threat"),
                    get_json_value(row, "insult"),
                    get_json_value(row, "identity_hate")
                });

                saved++;
            }

            progress_line("jigsaw", saved, offset);

            this_thread::sleep_for(chrono::milliseconds(REQUEST_DELAY_MS));
        }
        catch (const exception& error)
        {
            cout << "\nerror " << error.what() << endl;
            break;
        }
    }

    out.close();
    cout << endl;
    print_saved(output);
}

// downloads and saves the civil comments dataset
void fetch_civil_comments(const fs::path& out_dir, int max_rows)
{
    cout << "\nfetching civil comments" << endl;

    fs::path output = out_dir / "civil_comments.csv";
    ofstream out(output);

    if (!out.is_open())
    {
        throw runtime_error("could not create " + output.string());
    }

    write_csv_header(out, {
        "comment_text",
        "toxic",
        "toxicity",
        "severe_toxicity",
        "obscene",
        "threat",
        "insult",
        "identity_attack"
    });

    int saved = 0;

    for (int offset = 0; ; offset += PAGE_SIZE)
    {
        if (should_stop(saved, max_rows))
        {
            break;
        }

        try
        {
            string url = rows_url(
                "google/civil_comments",
                "default",
                "train",
                offset
            );

            string body = http_get(url);
            vector<string> rows = extract_row_objects(body);

            if (rows.empty())
            {
                break;
            }

            for (const string& row : rows)
            {
                if (should_stop(saved, max_rows))
                {
                    break;
                }

                string toxicity = get_json_value(row, "toxicity");
                string toxic = to_double(toxicity) >= 0.5 ? "1" : "0";

                write_csv_row(out, {
                    get_json_value(row, "text"),
                    toxic,
                    toxicity,
                    get_json_value(row, "severe_toxicity"),
                    get_json_value(row, "obscene"),
                    get_json_value(row, "threat"),
                    get_json_value(row, "insult"),
                    get_json_value(row, "identity_attack")
                });

                saved++;
            }

            progress_line("civil comments", saved, offset);

            this_thread::sleep_for(chrono::milliseconds(REQUEST_DELAY_MS));
        }
        catch (const exception& error)
        {
            cout << "\nerror " << error.what() << endl;
            break;
        }
    }

    out.close();
    cout << endl;
    print_saved(output);
}

// downloads and saves one twitter hate split
void fetch_twitter_split(
    ofstream& out,
    const string& split,
    int max_rows,
    int& total_saved
)
{
    int saved_for_split = 0;

    for (int offset = 0; ; offset += PAGE_SIZE)
    {
        if (should_stop(saved_for_split, max_rows))
        {
            break;
        }

        try
        {
            string url = rows_url(
                "cardiffnlp/tweet_eval",
                "hate",
                split,
                offset
            );

            string body = http_get(url);
            vector<string> rows = extract_row_objects(body);

            if (rows.empty())
            {
                break;
            }

            for (const string& row : rows)
            {
                if (should_stop(saved_for_split, max_rows))
                {
                    break;
                }

                write_csv_row(out, {
                    get_json_value(row, "text"),
                    get_json_value(row, "label"),
                    split
                });

                saved_for_split++;
                total_saved++;
            }

            progress_line("twitter hate " + split, total_saved, offset);

            this_thread::sleep_for(chrono::milliseconds(REQUEST_DELAY_MS));
        }
        catch (const exception& error)
        {
            cout << "\nerror in split "
                 << split
                 << " "
                 << error.what()
                 << endl;

            break;
        }
    }
}

// downloads and saves the twitter hate dataset
void fetch_twitter_hate(const fs::path& out_dir, int max_rows_per_split)
{
    cout << "\nfetching twitter hate speech" << endl;

    fs::path output = out_dir / "twitter_hate.csv";
    ofstream out(output);

    if (!out.is_open())
    {
        throw runtime_error("could not create " + output.string());
    }

    write_csv_header(out, {
        "comment_text",
        "toxic",
        "split"
    });

    int total_saved = 0;

    fetch_twitter_split(out, "train", max_rows_per_split, total_saved);
    fetch_twitter_split(out, "validation", max_rows_per_split, total_saved);
    fetch_twitter_split(out, "test", max_rows_per_split, total_saved);

    out.close();
    cout << endl;
    print_saved(output);
}

// prints a summary of all csv files
void summary(const fs::path& out_dir)
{
    long long total_rows = 0;

    cout << "\nsummary" << endl;

    for (const auto& item : fs::directory_iterator(out_dir))
    {
        if (!item.is_regular_file())
        {
            continue;
        }

        if (item.path().extension() != ".csv")
        {
            continue;
        }

        long long rows = count_csv_rows(item.path());
        double size_mb = static_cast<double>(fs::file_size(item.path())) / MB;

        cout << " "
             << item.path().filename().string()
             << " | "
             << rows
             << " rows | "
             << size_mb
             << " MB"
             << endl;

        total_rows += rows;
    }

    cout << " TOTAL: " << total_rows << " rows" << endl;
    cout << "\nFiles saved in: " << out_dir << "\n" << endl;
}

// prints help instructions
void print_help()
{
    cout << "NoBully dataset downloader\n\n";
    cout << "Usage:\n";
    cout << "  ./get_data              downloads all rows by default\n";
    cout << "  ./get_data --max 1000   downloads only 1000 rows per dataset or split\n";
    cout << "  ./get_data --max 0      downloads all rows\n";
    cout << "\n";
}

// reads the row limit from program arguments
int read_max_rows(int argc, char* argv[])
{
    if (argc >= 2)
    {
        string arg = argv[1];

        if (arg == "--help" || arg == "-h")
        {
            print_help();
            exit(0);
        }
    }

    if (argc >= 3 && string(argv[1]) == "--max")
    {
        return stoi(argv[2]);
    }

    return 0;
}

// starts the dataset download process
int main(int argc, char* argv[])
{
    try
    {
        int max_rows = read_max_rows(argc, argv);

        fs::path out_dir = fs::current_path() / "data";
        fs::create_directories(out_dir);

        cout << "Fetching datasets into: " << out_dir << "\n";

        if (max_rows == 0)
        {
            cout << "Mode: ALL rows by default\n";
        }
        else
        {
            cout << "Mode: max "
                 << max_rows
                 << " rows per dataset or split\n";
        }

        cout << "Page size: " << PAGE_SIZE << " rows per request\n";
        cout << "Required external program: curl\n";

        fetch_jigsaw(out_dir, max_rows);
        fetch_civil_comments(out_dir, max_rows);
        fetch_twitter_hate(out_dir, max_rows);

        summary(out_dir);
    }
    catch (const exception& error)
    {
        cerr << "fatal error: " << error.what() << endl;
        return 1;
    }

    return 0;
}