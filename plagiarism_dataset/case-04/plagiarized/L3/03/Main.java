public class Main {
    public static void main(String[] args) {
        int mile = 1;
        System.out.println("Miles\t\tKilometers");
        System.out.println("-------------------------------");
        while (mile <= 10) {
                System.out.println(mile + "\t\t" + mile * 1.609);
                mile++;
        }
    }
}
