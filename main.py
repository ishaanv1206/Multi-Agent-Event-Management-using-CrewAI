from crewai import Agent, Crew, Task, LLM
from pydantic import BaseModel
from crewai_tools import SerperDevTool, ScrapeWebsiteTool


serper_api_key = input("Enter your Serper API Key: ")
groq_api_key   = input("Enter your Groq API Key: ")
search_tool = SerperDevTool(api_key=serper_api_key)
scrape_tool = ScrapeWebsiteTool()

groq_llm = LLM(
    model="groq/llama-3.2-90b-text-preview",
    temperature=0.7,
    api_key=groq_api_key
)

# Agents
venue_coordinator = Agent(
    role="Venue Coordinator",
    goal="Identify and book an appropriate venue based on event requirements",
    tools=[search_tool, scrape_tool],
    backstory="With a keen sense of space and understanding of event logistics, you excel at finding and securing the perfect venue that fits the event's theme, size, and budget constraints.",
    llm=groq_llm,
    verbose=True
)

logistics_manager = Agent(
    role="Logistics Manager",
    goal="Manage all logistics for the event including catering and equipment",
    tools=[search_tool, scrape_tool],
    backstory="Organized and detail-oriented, you ensure that every logistical aspect of the event from catering to equipment setup is flawlessly executed.",
    llm=groq_llm,
    verbose=True
)

marketing_agent = Agent(
    role="Marketing and Communications Agent",
    goal="Effectively market the event and communicate with participants",
    tools=[search_tool, scrape_tool],
    backstory="Creative and communicative, you craft compelling messages and engage with potential attendees to maximize event exposure and participation.",
    llm=groq_llm,
    verbose=True
)


class VenueDetails(BaseModel):
    name: str
    address: str
    capacity: int
    booking_status: str

# Tasks
venue_task = Task(
    description="Find a venue in {event_city} that meets criteria for {event_topic}.",
    expected_output="All the details of a specifically chosen venue to accommodate the event.",
    human_input=True,
    output_json=VenueDetails,
    output_file="venue_details.json",
    agent=venue_coordinator
)

logistics_task = Task(
    description="Coordinate catering and equipment for an event with {expected_participants} participants on {tentative_date}.",
    expected_output="Confirmation of all logistics arrangements including catering and equipment setup.",
    human_input=True,
    async_execution=True,
    agent=logistics_manager
)

marketing_task = Task(
    description="Promote the {event_topic} aiming to engage at least {expected_participants} potential attendees.",
    expected_output="Report on marketing activities and attendee engagement formatted as markdown.",
    async_execution=True,
    output_file="marketing_report.md",
    agent=marketing_agent
)

# Crew setup
event_management_crew = Crew(
    agents=[venue_coordinator, logistics_manager, marketing_agent],
    tasks=[venue_task, logistics_task, marketing_task],
    verbose=True
)

# Event input
event_details = {
    'event_topic': "Tech Innovation Conference",
    'event_description': "A gathering of tech innovators and industry leaders to explore future technologies.",
    'event_city': "San Francisco",
    'tentative_date': "2024-09-15",
    'expected_participants': 500,
    'budget': 20000,
    'venue_type': "Conference Hall"
}

if __name__ == "__main__":
    result = event_management_crew.kickoff(inputs=event_details)

    import json
    from pprint import pprint

    with open("venue_details.json") as f:
        pprint(json.load(f))

    try:
        from IPython.display import Markdown
        print("\nMarketing Report:\n")
        print(Markdown("marketing_report.md"))
    except ImportError:
        with open("marketing_report.md") as f:
            print(f.read())
